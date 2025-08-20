#ifndef IPPL_ALVINE_MANAGER_H
#define IPPL_ALVINE_MANAGER_H

#include <memory>

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

template <typename T, unsigned Dim>
class AlvineManager
    : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                              LoadBalancer<T, Dim>> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t= FieldSolver<T, Dim>;
    using LoadBalancer_t= LoadBalancer<T, Dim>;
    using Base= ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;

protected:
    unsigned nt_m;
    unsigned it_m;
    Vector_t<int, Dim> nr_m;
    unsigned np_m;
    std::array<bool, Dim> decomp_m;
    bool isAllPeriodic_m;
    ippl::NDIndex<Dim> domain_m;
    std::string solver_m;
    int dump_freq_m;

public:
    AlvineManager(unsigned nt_, Vector_t<int, Dim>& nr_, unsigned np_, std::string& solver_, int dump_freq_)
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>, LoadBalancer<T, Dim>>() 
        , nt_m(nt_)
        , nr_m(nr_)
        , np_m(np_)
        , solver_m(solver_)
	, dump_freq_m(dump_freq_) {}

    ~AlvineManager(){}

protected:
    double time_m;
    double dt_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;
    Vector_t<double, Dim> origin_m;
    Vector_t<double, Dim> hr_m;

public:

    double getTime() { return time_m; }

    void setTime(double time_) { time_m = time_; }

    int getNt() const { return nt_m; }

    void setNt(int nt_) { nt_m = nt_; }

    virtual void dump() { /* default does nothing */ };

    void pre_step() override {
    }

    void post_step() override {
      Inform m("Step: ");
      this->time_m += this->dt_m;
      this->it_m++;

      if(this->it_m % dump_freq_m == 0) {
      	this->dump();
      }
      m << this->it_m << " Done" << endl;
    }

    void grid2par() override { 
	gatherCIC(); 
    }

    void gatherCIC() {
      this->pcontainer_m->P = 0.0;
      gather(this->pcontainer_m->P, this->fcontainer_m->getUField(), this->pcontainer_m->R);
    }

    void par2grid() override {
	scatterCIC(); 
    }

    void computeVelocityField() {

      VField_t<T, Dim> u_field = this->fcontainer_m->getUField();
      u_field = 0.0;

      if constexpr (Dim == 2) {
        const int nghost = u_field.getNghost();
        auto view = u_field.getView();

        auto omega_view = this->fcontainer_m->getOmegaField().getView();
        this->fcontainer_m->getOmegaField().fillHalo();

        Vector_t<double, Dim> hr = hr_m;
        Kokkos::parallel_for(
            "Assign rhs", ippl::getRangePolicy(view, nghost),
            KOKKOS_LAMBDA(const int i, const int j) {
                view(i, j) = {
                    (omega_view(i, j + 1) - omega_view(i, j - 1)) / (2 * hr(1)), 
                    -(omega_view(i + 1, j) - omega_view(i - 1, j)) / (2 * hr(0))
                };

            });
      } else if constexpr (Dim == 3) {
        //TODO compute velocity field in 3D, this should be a simple curl operation (one line)
      }
    }

    void scatterCIC() {
      this->fcontainer_m->getOmegaField() = 0.0;
      if constexpr (Dim == 2) {
          scatter(this->pcontainer_m->omega, this->fcontainer_m->getOmegaField(), this->pcontainer_m->R);
	  this->fcontainer_m->getOmegaField() = this->fcontainer_m->getOmegaField() / (hr_m[0] * hr_m[1]);
      } else if constexpr (Dim == 3) {
        //TODO: for some reason the scatter method doesn't work in three dimensions but gather does. 
      }
    }
};
#endif
