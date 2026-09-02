#include "Ippl.h"
#include <openPMD/openPMD.hpp>

#include <random>

template <class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout> {
    Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        this->addAttribute(Q);
    }

    ~Bunch() {}

    typedef ippl::ParticleAttrib<double> charge_container_type;
    charge_container_type Q;
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        using Mesh_t = ippl::UniformCartesian<double, 3>;
        typedef ippl::ParticleSpatialLayout<double, 3, Mesh_t> playout_type;
        typedef Bunch<playout_type> bunch_type;
        using Centering_t = Mesh_t::DefaultCentering;

        int pt = 128;
        ippl::Index I(pt);
        ippl::NDIndex<3> owned(I, I, I);

        std::array<bool, 3> isParallel;
        isParallel.fill(true);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<3> layout(MPI_COMM_WORLD, owned, isParallel);

        double dx                      = 1.0 / double(pt);
        ippl::Vector<double, 3> hx     = {dx, dx, dx};
        ippl::Vector<double, 3> origin = {0, 0, 0};
        Mesh_t mesh(owned, hx, origin);

        playout_type pl(layout, mesh);

        bunch_type bunch(pl);
        typedef ippl::Field<double, 3, Mesh_t, Centering_t> field_type;

        field_type field;

        field.initialize(mesh, layout);

        bunch.setParticleBC(ippl::BC::PERIODIC);

        int nRanks              = ippl::Comm->size();
        unsigned int nParticles = std::pow(128, 3);

        if (nParticles % nRanks > 0) {
            if (ippl::Comm->rank() == 0) {
                std::cerr << nParticles << " not a multiple of " << nRanks << std::endl;
            }
            return 0;
        }

        unsigned int nLoc = nParticles / nRanks;

        bunch.create(nLoc);

        std::mt19937_64 eng;
        eng.seed(42);
        eng.discard(nLoc * ippl::Comm->rank());
        std::uniform_real_distribution<double> unif(hx[0] / 2, 1 - (hx[0] / 2));

        typename bunch_type::particle_position_type::host_mirror_type R_host =
            bunch.R.getHostMirror();
        double sum_coord = 0.0;
        for (unsigned int i = 0; i < nLoc; ++i) {
            ippl::Vector<double, 3> r = {unif(eng), unif(eng), unif(eng)};
            R_host(i)                 = r;
            sum_coord += r[0] + r[1] + r[2];
        }
        Kokkos::deep_copy(bunch.R.getView(), R_host);

        double global_sum_coord = 0.0;
        ippl::Comm->reduce(sum_coord, global_sum_coord, 1, std::plus<double>());

        if (ippl::Comm->rank() == 0) {
            std::cout << "Sum coord: " << global_sum_coord << std::endl;
        }

        bunch.Q = 1.0;

        bunch.update();

        field = 0.0;

        scatter(bunch.Q, field, bunch.R);

        // Check charge conservation
        try {
            double Total_charge_field = field.sum();

            std::cout << "Total charge in the field:" << Total_charge_field << std::endl;
            std::cout << "Total charge of the particles:" << bunch.Q.sum() << std::endl;
            std::cout << "Error:" << std::fabs(bunch.Q.sum() - Total_charge_field) << std::endl;
        } catch (const std::exception& e) {
            std::cout << e.what() << std::endl;
        }

        // ============================================================
        // openPMD
        // ============================================================

        using namespace openPMD;

        Series series(
            "ippl_testscatter.h5",
            Access::CREATE_LINEAR,
            ippl::Comm->getCommunicator());
        //Series series(
        //    "ippl_testscatter.bp5",
        //    Access::CREATE_LINEAR,
        //    ippl::Comm->getCommunicator());
        series.setMeshesPath("fields");
        series.setParticlesPath("particles");

        // ------------------------------------------------------------
        // Iteration
        // ------------------------------------------------------------

        auto iteration = series.snapshots()[0];

        // ============================================================
        // PARTICLES
        // ============================================================

        auto particles = iteration.particles["bunch"];
        typename bunch_type::particle_position_type::host_mirror_type R_hostMirror =
            bunch.R.getHostMirror();
        //typename ippl::ParticleAttrib<double>::host_mirror_type Q_hostMirror = bunch.Q.getHostMirror();
        Kokkos::deep_copy(R_hostMirror, bunch.R.getView());
        //Kokkos::deep_copy(Q_hostMirror, bunch.Q.getView());

        const size_t localNum = bunch.getLocalNum();
        const int rank = ippl::Comm->rank();

        using HostParticleView_t = Kokkos::View<double*, Kokkos::HostSpace>;
        HostParticleView_t Rx("Rx", localNum);
        HostParticleView_t Ry("Ry", localNum);
        HostParticleView_t Rz("Rz", localNum);
        using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
        if (localNum > 0) {
            Kokkos::RangePolicy<HostExecSpace> host_policy(0, localNum);
            Kokkos::parallel_for("AoS to SoA", host_policy, KOKKOS_LAMBDA(const int64_t i) {
                Rx(i) = R_hostMirror(i)[0];
                Ry(i) = R_hostMirror(i)[1];
                Rz(i) = R_hostMirror(i)[2];
            });
        }
        // ------------------------------------------------------------
        // Global particle dataset
        // ------------------------------------------------------------

        Dataset particle_dataset(determineDatatype<double>(),{static_cast<std::size_t>(nParticles)});
        auto positionOffset = particles["positionOffset"];

        particles["position"]["x"].resetDataset(particle_dataset);
        particles["position"]["y"].resetDataset(particle_dataset);
        particles["position"]["z"].resetDataset(particle_dataset);
        particles["weighting"].resetDataset(particle_dataset);
        positionOffset["x"].resetDataset(particle_dataset).makeConstant(0.0);
        positionOffset["y"].resetDataset(particle_dataset).makeConstant(0.0);
        positionOffset["z"].resetDataset(particle_dataset).makeConstant(0.0);

        std::size_t offset = 0;
        MPI_Exscan(&localNum,&offset,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,ippl::Comm->getCommunicator());
        
        if (rank == 0)
            offset = 0;

        Offset particle_offset = {static_cast<std::size_t>(offset)};
        Extent particle_extent = {static_cast<std::size_t>(localNum)};

        particles["position"]["x"].storeChunkRaw(
            Rx.data(),
            particle_offset,
            particle_extent);

        particles["position"]["y"].storeChunkRaw(
            Ry.data(),
            particle_offset,
            particle_extent);

        particles["position"]["z"].storeChunkRaw(
            Rz.data(),
            particle_offset,
            particle_extent);

        //particles["weighting"].storeChunkRaw(
        //    Q_hostMirror.data(),
        //    particle_offset,
        //    particle_extent);

        particles["weighting"].storeChunkRaw(
            bunch.Q.getView().data(),
            particle_offset,
            particle_extent);

        
        // ============================================================
        // FIELD
        // ============================================================
        
        auto rho = iteration.meshes["charge_density"];
        
        rho.setGeometry(openPMD::Mesh::Geometry::cartesian);
        rho.setDataOrder(openPMD::Mesh::DataOrder::C);
        
        rho.setGridSpacing(std::vector<double>{dx, dx, dx});
        rho.setGridGlobalOffset(std::vector<double>{0.0, 0.0, 0.0});
        
        // ------------------------------------------------------------
        // IPPL local layout
        // ------------------------------------------------------------
        
        auto& fieldLayout = field.getLayout();
        auto localNDIndex = fieldLayout.getLocalNDIndex();
        
        const auto& fullView = field.getView();
        
        const std::size_t nGhost = field.getNghost();
        
        // Number of owned cells in each direction
        const std::size_t nx = localNDIndex[0].length();
        const std::size_t ny = localNDIndex[1].length();
        const std::size_t nz = localNDIndex[2].length();
        
        // ------------------------------------------------------------
        // Global offset of this rank's local field
        //
        // localNDIndex gives the indices in the global field.
        // ------------------------------------------------------------
        
        const std::size_t ox =
            static_cast<std::size_t>(localNDIndex[0].first());
        
        const std::size_t oy =
            static_cast<std::size_t>(localNDIndex[1].first());
        
        const std::size_t oz =
            static_cast<std::size_t>(localNDIndex[2].first());
        
        // ------------------------------------------------------------
        // Extract the owned region.
        //
        // The IPPL field view includes ghost cells, so skip nGhost
        // cells on every side.
        //
        // We use a host mirror for this first prototype.
        // ------------------------------------------------------------
        
        using FieldView =
            typename field_type::view_type;
        
        using HostView =
            Kokkos::View<
                typename FieldView::data_type,
                Kokkos::LayoutRight,
                Kokkos::HostSpace>;

        auto rho_host_full = Kokkos::create_mirror_view(fullView);
        Kokkos::deep_copy(rho_host_full, fullView);
        
        HostView rho_host(
            "rho_host",
            nx, ny, nz);
        
        //auto r0 = Kokkos::make_pair(
        //    nGhost,
        //    nGhost + nx);
        //
        //auto r1 = Kokkos::make_pair(
        //    nGhost,
        //    nGhost + ny);
        //
        //auto r2 = Kokkos::make_pair(
        //    nGhost,
        //    nGhost + nz);
        //
        //auto rho_subview =
        //    Kokkos::subview(
        //        fullView,
        //        r0, r1, r2);
        //
        //Kokkos::deep_copy(
        //    rho_host,
        //    rho_subview);
        
        // host → host copy, potentially doing the layout conversion

        Kokkos::parallel_for("extract_rho",
            Kokkos::MDRangePolicy<HostExecSpace,Kokkos::Rank<3>>(
            {0, 0, 0}, {nx, ny, nz}),
            KOKKOS_LAMBDA(int i, int j, int k) {
                rho_host(i, j, k) =
                    rho_host_full(i + nGhost,
                                  j + nGhost,
                                  k + nGhost);
            });
        
        // ------------------------------------------------------------
        // Global mesh dataset
        // ------------------------------------------------------------
        
        const std::size_t globalNx = pt;
        const std::size_t globalNy = pt;
        const std::size_t globalNz = pt;
        
        openPMD::Dataset rho_dataset(
            openPMD::Datatype::DOUBLE,
            {globalNx, globalNy, globalNz});
        
        rho["SCALAR"].resetDataset(rho_dataset);
        
        // ------------------------------------------------------------
        // Write this rank's local chunk
        // ------------------------------------------------------------
        
        rho["SCALAR"].storeChunkRaw(
            rho_host.data(),
            {ox, oy, oz},
            {nx, ny, nz});

        // ------------------------------------------------------------
        // Close/write
        // ------------------------------------------------------------
        series.flush();
        iteration.close();
        series.close();
    }
    ippl::finalize();
    return 0;
}
