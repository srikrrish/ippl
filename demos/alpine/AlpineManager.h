#ifndef IPPL_ALPINE_MANAGER_H
#define IPPL_ALPINE_MANAGER_H

#include <memory>
#include <sstream>
#include <vector>

#include <openPMD/openPMD.hpp>

#include "FEM/FEMInterpolate.hpp"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "Manager/PicManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

template <typename VectorAttrib>
void aos_to_soa(const VectorAttrib& attrib,
                Kokkos::View<double*, Kokkos::HostSpace>& Fx,
                Kokkos::View<double*, Kokkos::HostSpace>& Fy,
                Kokkos::View<double*, Kokkos::HostSpace>& Fz,
                std::size_t localNum)
{
    // Mirror device view to host
    auto mirror = attrib.getHostMirror();
    Kokkos::deep_copy(mirror, attrib.getView());

    Fx = Kokkos::View<double*, Kokkos::HostSpace>("Fx", localNum);
    Fy = Kokkos::View<double*, Kokkos::HostSpace>("Fy", localNum);
    Fz = Kokkos::View<double*, Kokkos::HostSpace>("Fz", localNum);

    using HostExec = Kokkos::DefaultHostExecutionSpace;
    Kokkos::parallel_for(
        "aos_to_soa",
        Kokkos::RangePolicy<HostExec>(0, localNum),
        KOKKOS_LAMBDA(const int64_t i) {
            Fx(i) = mirror(i)[0];
            Fy(i) = mirror(i)[1];
            Fz(i) = mirror(i)[2];
        });
    Kokkos::fence();
}

template <typename T, unsigned Dim>
class AlpineManager : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>,
                                              FieldContainer<T, Dim>, LoadBalancer<T, Dim>> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;
    using Base =
        ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim, ippl::UniformCartesian<T, Dim>>>;

protected:
    size_type totalP_m;
    int nt_m;
    Vector_t<int, Dim> nr_m;
    double lbt_m;
    std::string solver_m;
    std::string stepMethod_m;
    std::vector<std::string> preconditioner_params_m;

public:
    AlpineManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                  std::string& solver_, std::string& stepMethod_,
                  std::vector<std::string> preconditioner_params = {})
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                           LoadBalancer<T, Dim>>()
        , totalP_m(totalP_)
        , nt_m(nt_)
        , nr_m(nr_)
        , lbt_m(lbt_)
        , solver_m(solver_)
        , stepMethod_m(stepMethod_)
        , preconditioner_params_m(preconditioner_params) {}
    ~AlpineManager() { closeOpenPMD(); }

protected:
    double time_m;
    double dt_m;
    int it_m;
    Vector_t<double, Dim> kw_m;
    double alpha_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;
    Vector_t<double, Dim> hr_m;
    double Q_m;
    Vector_t<double, Dim> origin_m;
    bool isAllPeriodic_m;
    bool isFirstRepartition_m;
    ippl::NDIndex<Dim> domain_m;
    std::array<bool, Dim> decomp_m;
    double rhoNorm_m;

    // ------------------------------------------------------------------
    // openPMD: particle-only output (position x,y,z  +  E x,y,z)
    // ------------------------------------------------------------------
    std::unique_ptr<openPMD::Series> openpmd_series_m;
    std::string                      openpmd_species_m = "bunch";

    // Call once after particles and fields are initialised (end of pre_run).
    // filename  – e.g. "LandauDamping.h5"  or  "LandauDamping.bp5"
    // species   – label under particles/<species>/…  (default: "bunch")
    void openOpenPMD(const std::string& filename,
                     const std::string& species = "bunch")
    {
        openpmd_species_m = species;
        openpmd_series_m  = std::make_unique<openPMD::Series>(
            filename,
            openPMD::Access::CREATE_LINEAR,
            ippl::Comm->getCommunicator());
        openpmd_series_m->setMeshesPath("fields");
        openpmd_series_m->setParticlesPath("particles");
    }

    // Call once after the time loop (or rely on the destructor).
    void closeOpenPMD()
    {
        if (!openpmd_series_m) return;
        openpmd_series_m->flush();
        openpmd_series_m->close();
        openpmd_series_m.reset();
    }

    // Write one snapshot: particle positions (inputs) + gathered E (outputs).
    // Must be called after grid2par() so pcontainer_m->E is populated.
    void dumpOpenPMD()
    {
        if (!openpmd_series_m) return;

        auto& pc       = *this->pcontainer_m;
        const auto localNum  = static_cast<std::size_t>(pc.getLocalNum());
        const auto totalNum  = static_cast<std::size_t>(totalP_m);
        const int  rank      = ippl::Comm->rank();

        // ---- AoS (device) → SoA (host) via free-function helper -----------
        // aos_to_soa mirrors device→host and runs a Kokkos::parallel_for on
        // the host execution space. Being a free function it has no class
        // access restrictions, so CUDA accepts the KOKKOS_LAMBDA inside it.
        using HostView = Kokkos::View<double*, Kokkos::HostSpace>;
        HostView Rx, Ry, Rz, Ex, Ey, Ez;
        aos_to_soa(pc.R, Rx, Ry, Rz, localNum);
        aos_to_soa(pc.E, Ex, Ey, Ez, localNum);

        // ---- MPI offset (exclusive prefix sum of local counts) ------------
        std::size_t offset = 0;
        MPI_Exscan(&localNum, &offset, 1,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                   ippl::Comm->getCommunicator());
        if (rank == 0) offset = 0;

        openPMD::Offset pOff = {offset};
        openPMD::Extent pExt = {localNum};
        openPMD::Dataset pDS(openPMD::determineDatatype<double>(), {totalNum});

        // ---- openPMD snapshot --------------------------------------------
        auto iteration = openpmd_series_m->snapshots()[it_m];
        iteration.setTime(time_m);

        auto particles = iteration.particles[openpmd_species_m];

        // positionOffset (required by standard)
        auto posOff = particles["positionOffset"];
        posOff["x"].resetDataset(pDS).makeConstant(0.0);
        posOff["y"].resetDataset(pDS).makeConstant(0.0);
        posOff["z"].resetDataset(pDS).makeConstant(0.0);

        // position – training INPUTS
        particles["position"]["x"].resetDataset(pDS);
        particles["position"]["y"].resetDataset(pDS);
        particles["position"]["z"].resetDataset(pDS);
        particles["position"]["x"].storeChunkRaw(Rx.data(), pOff, pExt);
        particles["position"]["y"].storeChunkRaw(Ry.data(), pOff, pExt);
        particles["position"]["z"].storeChunkRaw(Rz.data(), pOff, pExt);

        // gathered E field – training OUTPUTS
        particles["E"]["x"].resetDataset(pDS);
        particles["E"]["y"].resetDataset(pDS);
        particles["E"]["z"].resetDataset(pDS);
        particles["E"]["x"].storeChunkRaw(Ex.data(), pOff, pExt);
        particles["E"]["y"].storeChunkRaw(Ey.data(), pOff, pExt);
        particles["E"]["z"].storeChunkRaw(Ez.data(), pOff, pExt);

        iteration.close();   // flush this snapshot; buffers are released
    }

public:
    size_type getTotalP() const { return totalP_m; }

    void setTotalP(size_type totalP_) { totalP_m = totalP_; }

    int getNt() const { return nt_m; }

    void setNt(int nt_) { nt_m = nt_; }

    const std::string& getSolver() const { return solver_m; }

    void setSolver(const std::string& solver_) { solver_m = solver_; }

    double getLoadBalanceThreshold() const { return lbt_m; }

    void setLoadBalanceThreshold(double lbt_) { lbt_m = lbt_; }

    const std::string& getStepMethod() const { return stepMethod_m; }

    void setStepMethod(const std::string& stepMethod_) { stepMethod_m = stepMethod_; }

    const Vector_t<int, Dim>& getNr() const { return nr_m; }

    void setNr(const Vector_t<int, Dim>& nr_) { nr_m = nr_; }

    double getTime() const { return time_m; }

    void setTime(double time_) { time_m = time_; }

    std::vector<std::string> getPreconditionerParams() const { return preconditioner_params_m; };

    virtual void dump() { /* default does nothing */ };

    void pre_step() override {
        Inform m("Pre-step");
        m << "Done" << endl;
    }

    void post_step() override {
        // Update time
        this->time_m += this->dt_m;
        this->it_m++;
        // write solution to output file
        this->dump();

        Inform m("Post-step:");
        m << "Finished time step: " << this->it_m << " time: " << this->time_m << endl;
    }

    void grid2par() override {
        if ((getSolver() == "FEM") || (getSolver() == "FEM_PRECON")) {
            gatherFEM();
        } else {
            gatherCIC();
        }
    }

    void gatherCIC() {
        gather(this->pcontainer_m->E, this->fcontainer_m->getE(), this->pcontainer_m->R);
    }

    void gatherFEM() {
        using exec_space         = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type        = Kokkos::RangePolicy<exec_space>;
        size_type localParticles = this->pcontainer_m->getLocalNum();
        policy_type iteration_policy(0, localParticles);

        // get the Finite Element space from the solver,
        // since the interpolation depends on the space
        auto* solver = dynamic_cast<FieldSolver_t*>(this->fsolver_m.get());
        auto& space  = solver->getSpace();

        interpolate_grad_to_diracs(this->pcontainer_m->E, this->fcontainer_m->getPhi(),
                                   this->pcontainer_m->R, space, iteration_policy);
    }

    void par2grid() override {
        if ((getSolver() == "FEM") || (getSolver() == "FEM_PRECON")) {
            scatterFEM();
        } else {
            scatterCIC();
        }
    }

    void scatterCIC() {
        Inform m("scatter ");

        this->fcontainer_m->getRho() = 0.0;

        ippl::ParticleAttrib<double>* q          = &this->pcontainer_m->q;
        typename Base::particle_position_type* R = &this->pcontainer_m->R;
        Field_t<Dim>* rho                        = &this->fcontainer_m->getRho();
        double Q                                 = Q_m;

        scatter(*q, *rho, *R);

        double relError = std::fabs((Q - (*rho).sum()) / Q);
        m << relError << endl;

        checkChargeConservation(relError, m);

        getDensity(rho);
    }

    void scatterFEM() {
        Inform m("scatter ");

        this->fcontainer_m->getRho() = 0.0;

        ippl::ParticleAttrib<double>* q          = &this->pcontainer_m->q;
        typename Base::particle_position_type* R = &this->pcontainer_m->R;
        Field_t<Dim>* rho                        = &this->fcontainer_m->getRho();
        double Q                                 = Q_m;
        size_type localParticles                 = this->pcontainer_m->getLocalNum();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;
        policy_type iteration_policy(0, localParticles);

        // get the Finite Element space from the solver,
        // since the interpolation depends on the space
        auto* solver = dynamic_cast<FieldSolver_t*>(this->fsolver_m.get());
        auto& space  = solver->getSpace();

        assemble_rhs_from_particles(*q, *rho, *R, space, iteration_policy);

        double relError = std::fabs((Q - (*rho).sum()) / Q);
        m << relError << endl;

        double num = 1e-14;
        checkChargeConservation(num, m);

        getDensity(rho);
    }

    void checkChargeConservation(double& relError, Inform& m) {
        size_type TotalParticles = 0;
        size_type localParticles = this->pcontainer_m->getLocalNum();

        ippl::Comm->reduce(localParticles, TotalParticles, 1, std::plus<size_type>());

        if (ippl::Comm->rank() == 0) {
            if (TotalParticles != totalP_m || relError > 1e-10) {
                m << "Time step: " << it_m << endl;
                m << "Total particles in the sim. " << totalP_m << " "
                  << "after update: " << TotalParticles << endl;
                m << "Rel. error in charge conservation: " << relError << endl;
                ippl::Comm->abort();
            }
        }
    }

    void getDensity(Field_t<Dim>* rho) {
        Vector_t<double, Dim> rmin = rmin_m;
        Vector_t<double, Dim> rmax = rmax_m;
        Vector_t<double, Dim> hr   = this->hr_m;
        double Q                   = Q_m;

        if ((this->fsolver_m->getStype() != "FEM")
            && (this->fsolver_m->getStype() != "FEM_PRECON")) {
            double cellVolume = std::reduce(hr.begin(), hr.end(), 1., std::multiplies<double>());
            (*rho)            = (*rho) / cellVolume;
        }

        // rho = rho_e - rho_i (only if periodic BCs)
        if (this->fsolver_m->getStype() != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= rmax[d] - rmin[d];
            }
            *rho = *rho - (Q / size);
        }
    }
};
#endif
