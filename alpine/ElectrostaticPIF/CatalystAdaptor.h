#ifndef CatalystAdaptor_h
#define CatalystAdaptor_h

#include "Ippl.h"

#include <catalyst.hpp>
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <variant>
#include <utility>

#include "Utility/IpplException.h"


namespace CatalystAdaptor {

    template <typename T, unsigned Dim>
    using FieldVariant = std::variant<Field_t<Dim>*, VField_t<T, Dim>*>;

    template <typename T, unsigned Dim>
    using FieldPair = std::pair<std::string, FieldVariant<T, Dim>>;

    //template <typename T, unsigned Dim>
    //using ParticlePair = std::pair<std::string, std::shared_ptr<ParticleContainer<T, Dim> > >;


    using View_vector =
        Kokkos::View<ippl::Vector<double, 3>***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    inline void setData(conduit_cpp::Node& node, const View_vector& view) {
        node["electrostatic/association"].set_string("element");
        node["electrostatic/topology"].set_string("mesh");
        node["electrostatic/volume_dependent"].set_string("false");

        auto length = std::size(view);

        // offset is zero as we start without the ghost cells
        // stride is 1 as we have every index of the array
        node["electrostatic/values/x"].set_external(&view.data()[0][0], length, 0, 1);
        node["electrostatic/values/y"].set_external(&view.data()[0][1], length, 0, 1);
        node["electrostatic/values/z"].set_external(&view.data()[0][2], length, 0, 1);
    }


    using View_scalar = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    inline void setData(conduit_cpp::Node& node, const View_scalar& view) {
        node["density/association"].set_string("element");
        node["density/topology"].set_string("mesh");
        node["density/volume_dependent"].set_string("false");

        node["density/values"].set_external(view.data(), view.size());
    }


    void Initialize(int /*argc*/, char* argv[]) {
        conduit_cpp::Node node;
        std::cout << "pvscript path: " << argv[1] << std::endl;
        //std::cout << "pvproxy path: " << argv[2] << std::endl;
        node["catalyst/scripts/script/filename"].set(argv[1]);
        //node["catalyst/proxies/proxy"].set(argv[2]);
        /*
        for (int cc = 2; cc < argc; ++cc) {
            std::cout << "pvscript args: " << argv[cc] << std::endl;
            conduit_cpp::Node list_entry = node["catalyst/scripts/script/args"].append();
            list_entry.set(argv[cc]);
        }
        */
        try {
            node["catalyst_load/implementation"]        = getenv("CATALYST_IMPLEMENTATION_NAME");
            node["catalyst_load/search_paths/paraview"] = getenv("CATALYST_IMPLEMENTATION_PATHS");
        } catch (...) {
            throw IpplException("CatalystAdaptor::Initialize",
                                "no environmental variable for CATALYST_IMPLEMENTATION_NAME or "
                                "CATALYST_IMPLEMENTATION_PATHS found");
        }
        // TODO: catch catalyst error also with IpplException
        catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to initialize Catalyst: " << err << std::endl;
        }
    }


    void Finalize() {
        conduit_cpp::Node node;
        catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to finalize Catalyst: " << err << std::endl;
        }
    }


    void Execute_Particle(
         const auto& particleContainer,
         const auto& R_host, const auto& P_host, const auto& q_host, const auto& ID_host,
         conduit_cpp::Node& node) {

        // channel for particles
        auto channel = node["catalyst/channels/ippl_" + particlesName];
        channel["type"].set_string("mesh");

        // in data channel now we adhere to conduits mesh blueprint definition
        auto mesh = channel["data"];
        mesh["coordsets/coords/type"].set_string("explicit");

        //mesh["coordsets/coords/values/x"].set_external(&layout_view.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //mesh["coordsets/coords/values/y"].set_external(&layout_view.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //mesh["coordsets/coords/values/z"].set_external(&layout_view.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        mesh["coordsets/coords/values/x"].set(&R_host.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        mesh["coordsets/coords/values/y"].set(&R_host.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        mesh["coordsets/coords/values/z"].set(&R_host.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);

        mesh["topologies/mesh/type"].set_string("unstructured");
        mesh["topologies/mesh/coordset"].set_string("coords");
        mesh["topologies/mesh/elements/shape"].set_string("point");
        //mesh["topologies/mesh/elements/connectivity"].set_external(particleContainer->ID.getView().data(),particleContainer->getLocalNum());
        mesh["topologies/mesh/elements/connectivity"].set(ID_host.data(),particleContainer->getLocalNum());

        //auto charge_view = particleContainer->getQ().getView();

        // add values for scalar charge field
        auto fields = mesh["fields"];
        fields["charge/association"].set_string("vertex");
        fields["charge/topology"].set_string("mesh");
        fields["charge/volume_dependent"].set_string("false");

        //fields["charge/values"].set_external(particleContainer->q.getView().data(), particleContainer->getLocalNum());
        fields["charge/values"].set(q_host.data(), particleContainer->getLocalNum());

        // add values for vector velocity field
        //auto velocity_view = particleContainer->P.getView();
        fields["velocity/association"].set_string("vertex");
        fields["velocity/topology"].set_string("mesh");
        fields["velocity/volume_dependent"].set_string("false");

        //fields["velocity/values/x"].set_external(&velocity_view.data()[0][0], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        //fields["velocity/values/y"].set_external(&velocity_view.data()[0][1], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        //fields["velocity/values/z"].set_external(&velocity_view.data()[0][2], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        fields["velocity/values/x"].set(&P_host.data()[0][0], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        fields["velocity/values/y"].set(&P_host.data()[0][1], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        fields["velocity/values/z"].set(&P_host.data()[0][2], particleContainer->getLocalNum(),0 ,sizeof(double)*3);

        fields["position/association"].set_string("vertex");
        fields["position/topology"].set_string("mesh");
        fields["position/volume_dependent"].set_string("false");

        //fields["position/values/x"].set_external(&layout_view.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //fields["position/values/y"].set_external(&layout_view.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //fields["position/values/z"].set_external(&layout_view.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        fields["position/values/x"].set(&R_host.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        fields["position/values/y"].set(&R_host.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        fields["position/values/z"].set(&R_host.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);
    }


    template <class Field>  // == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
    void Execute_Field(Field* field, const std::string& fieldName,
         Kokkos::View<typename Field::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>& host_view_layout_left,
         conduit_cpp::Node& node) {
        static_assert(Field::dim == 3, "CatalystAdaptor only supports 3D");

        // A) define mesh

        // add catalyst channel named ippl_"field", as fields is reserved
        auto channel = node["catalyst/channels/ippl_" + fieldName];
        channel["type"].set_string("mesh");

        // in data channel now we adhere to conduits mesh blueprint definition
        auto mesh = channel["data"];
        mesh["coordsets/coords/type"].set_string("uniform");

        // number of points in specific dimension
        std::string field_node_dim{"coordsets/coords/dims/i"};
        std::string field_node_origin{"coordsets/coords/origin/x"};
        std::string field_node_spacing{"coordsets/coords/spacing/dx"};

        for (unsigned int iDim = 0; iDim < field->get_mesh().getGridsize().dim; ++iDim) {
            // add dimension
            mesh[field_node_dim].set(field->getLayout().getLocalNDIndex()[iDim].length() + 1);

            // add origin
            mesh[field_node_origin].set(
                field->get_mesh().getOrigin()[iDim] + field->getLayout().getLocalNDIndex()[iDim].first()
                      * field->get_mesh().getMeshSpacing(iDim));

            // add spacing
            mesh[field_node_spacing].set(field->get_mesh().getMeshSpacing(iDim));

            // increment last char in string
            ++field_node_dim.back();
            ++field_node_origin.back();
            ++field_node_spacing.back();
        }

        // add topology
        mesh["topologies/mesh/type"].set_string("uniform");
        mesh["topologies/mesh/coordset"].set_string("coords");
        std::string field_node_origin_topo{"topologies/mesh/origin/x"};
        for (unsigned int iDim = 0; iDim < field->get_mesh().getGridsize().dim; ++iDim) {
            // shift origin
            mesh[field_node_origin_topo].set(field->get_mesh().getOrigin()[iDim]
                                             + field->getLayout().getLocalNDIndex()[iDim].first()
                                                   * field->get_mesh().getMeshSpacing(iDim));

            // increment last char in string ('x' becomes 'y' becomes 'z')
            ++field_node_origin_topo.back();
        }

        // B) Set the field values

        // Initialize the existing Kokkos::View
        host_view_layout_left = Kokkos::View<typename Field::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>(
           "host_view_layout_left",
           field->getLayout().getLocalNDIndex()[0].length(),
           field->getLayout().getLocalNDIndex()[1].length(),
           field->getLayout().getLocalNDIndex()[2].length());

        // Creates a host-accessible mirror view and copies the data from the device view to the host.
        auto host_view =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field->getView());

        // Copy data from field to the memory+style which will be passed to Catalyst
        auto nGhost = field->getNghost();
        for (size_t i = 0; i < field->getLayout().getLocalNDIndex()[0].length(); ++i) {
            for (size_t j = 0; j < field->getLayout().getLocalNDIndex()[1].length(); ++j) {
                for (size_t k = 0; k < field->getLayout().getLocalNDIndex()[2].length(); ++k) {
                    host_view_layout_left(i, j, k) = host_view(i + nGhost, j + nGhost, k + nGhost);
                }
            }
        }

        // Add values and subscribe to data
        auto fields = mesh["fields"];
        setData(fields, host_view_layout_left);
    }

    void AddSteerableChannel(conduit_cpp::Node& node, double scaleFactorE, double scaleFactorB) {
        auto steerable = node["catalyst/channels/steerableE"];
        steerable["type"].set("mesh");

        auto steerable_data = steerable["data"];
        steerable_data["coordsets/coords/type"].set_string("explicit");
        steerable_data["coordsets/coords/values/x"].set_float64_vector({ 1 });
        steerable_data["coordsets/coords/values/y"].set_float64_vector({ 2 });
        steerable_data["coordsets/coords/values/z"].set_float64_vector({ 3 });
        steerable_data["topologies/mesh/type"].set("unstructured");
        steerable_data["topologies/mesh/coordset"].set("coords");
        steerable_data["topologies/mesh/elements/shape"].set("point");
        steerable_data["topologies/mesh/elements/connectivity"].set_int32_vector({ 0 });
        steerable_data["fields/steerableE/association"].set("vertex");
        steerable_data["fields/steerableE/topology"].set("mesh");
        steerable_data["fields/steerableE/volume_dependent"].set("false");
        steerable_data["fields/steerableE/values"].set_float64_vector({ scaleFactorE });
        steerable_data["fields/steerableB/association"].set("vertex");
        steerable_data["fields/steerableB/topology"].set("mesh");
        steerable_data["fields/steerableB/volume_dependent"].set("false");
        steerable_data["fields/steerableB/values"].set_float64_vector({scaleFactorB});
        //steerable_data["fields/steerable/values/y"].set_float64_vector({ scaleFactorB });
        //steerable_data["fields/steerable/values/z"].set_float64_vector({ 0 });
        


        auto steerableB = node["catalyst/channels/steerableB"];
        steerableB["type"].set("mesh");

        auto steerable_dataB = steerableB["data"];
        steerable_dataB["coordsets/coords/type"].set_string("explicit");
        steerable_dataB["coordsets/coords/values/x"].set_float64_vector({ 1 });
        steerable_dataB["coordsets/coords/values/y"].set_float64_vector({ 2 });
        steerable_dataB["coordsets/coords/values/z"].set_float64_vector({ 3 });
        steerable_dataB["topologies/mesh/type"].set("unstructured");
        steerable_dataB["topologies/mesh/coordset"].set("coords");
        steerable_dataB["topologies/mesh/elements/shape"].set("point");
        steerable_dataB["topologies/mesh/elements/connectivity"].set_int32_vector({ 0 });
        steerable_dataB["fields/steerableB/association"].set("vertex");
        steerable_dataB["fields/steerableB/topology"].set("mesh");
        steerable_dataB["fields/steerableB/volume_dependent"].set("false");
        steerable_dataB["fields/steerableB/values"].set_float64_vector({scaleFactorB});
    }

    void Results( double& scaleFactorE, double& scaleFactorB) {
        
        conduit_cpp::Node results;//, resultsB;
        catalyst_status err = catalyst_results(conduit_cpp::c_node(&results));
        //catalyst_status errB = catalyst_results(conduit_cpp::c_node(&resultsB));

        //if ((errE != catalyst_status_ok) || (errB != catalyst_status_ok))
        if (err != catalyst_status_ok)
        {
            std::cerr << "Failed to execute Catalyst-results: " << err << std::endl;
        }
        else
        {
            std::cout << "Result Node dump:" << std::endl;
            const std::string value_pathE = "catalyst/steerable/fields/scalefactorE/values";
            //const std::string value_pathB = "catalyst/steerable/fields/scaleFactor/values/y";
            const std::string value_pathB = "catalyst/steerable/fields/scaleFactorB/values";
            if (results.has_path(value_pathE)) {
                scaleFactorE = results[value_pathE].to_double();
            }
            else 
            {
                std::cerr << "key: [" << value_pathE << "] not found!" << std::endl;
            }
            if (results.has_path(value_pathB)) {
                scaleFactorB = results[value_pathB].to_double();
            }
            else 
            {
                std::cerr << "key: [" << value_pathB << "] not found!" << std::endl;
            }
            //auto nodeE = results[value_pathE].as_float64_ptr();
            //auto nodeB = results[value_pathB].as_float64_ptr();
            //scaleFactorE = nodeE[0];
            //scaleFactorB = nodeB[0];
            //scaleFactorB = resultsB[value_pathB].to_double();
        }   
    }

    template <typename T, unsigned Dim, class PLayout>
    void Execute(int cycle, double time, int rank,
    const std::shared_ptr<ChargedParticlesPIF<PLayout>>& particleContainer,
    //const std::vector<CatalystAdaptor::ParticlePair<T, Dim>>& particles,
    const std::vector<FieldPair<T, Dim>>& fields) {

        // catalyst blueprint definition
        // https://docs.paraview.org/en/latest/Catalyst/blueprints.html
        //
        // conduit blueprint definition (v.8.3)
        // https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html
        conduit_cpp::Node node;

        // add time/cycle information
        auto state = node["catalyst/state"];
        state["cycle"].set(cycle);
        state["time"].set(time);
        state["domain_id"].set(rank);

        // Handle particles

        typename ippl::ParticleAttrib<ippl::Vector<double, 3>::HostMirror R_host_map;
        typename ippl::ParticleAttrib<ippl::Vector<double, 3>::HostMirror P_host_map;
        typename ippl::ParticleAttrib<double>::HostMirror q_host_map;
        typename ippl::ParticleAttrib<std::int64_t>::HostMirror ID_host_map;


        assert((particleContainer->ID.getView().data() != nullptr) && "ID view should not be nullptr, might be missing the right execution space");

        R_host_map  = particleContainer->R.getHostMirror();
        P_host_map  = particleContainer->P.getHostMirror();
        q_host_map  = particleContainer->q.getHostMirror();
        ID_host_map = particleContainer->ID.getHostMirror();

        Kokkos::deep_copy(R_host_map,  particleContainer->R.getView());
        Kokkos::deep_copy(P_host_map,  particleContainer->P.getView());
        Kokkos::deep_copy(q_host_map,  particleContainer->q.getView());
        Kokkos::deep_copy(ID_host_map, particleContainer->ID.getView());

        Execute_Particle(
          particleContainer,
          R_host_map, P_host_map, q_host_map, ID_host_map,
          node);


        // Handle fields

        // Map of all Kokkos::Views. This keeps a reference on all Kokkos::Views
        // which ensures that Kokkos does not free the memory before the end of this function.
        std::map<std::string, Kokkos::View<typename Field_t<Dim>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace> > scalar_host_views;
        std::map<std::string, Kokkos::View<typename VField_t<T, Dim>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace> > vector_host_views;

        // Loop over all fields
        for (const auto& fieldPair : fields)
        {
            const std::string& fieldName = fieldPair.first;
            const auto& fieldVariant = fieldPair.second;

            // If field is a _scalar_ field
            if (std::holds_alternative<Field_t<Dim>*>(fieldVariant)) {
                Field_t<Dim>* field = std::get<Field_t<Dim>*>(fieldVariant);
                // == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*

                Execute_Field(field, fieldName, scalar_host_views[fieldName], node);
            }
            // If field is a _vector_ field
            else if (std::holds_alternative<VField_t<T, Dim>*>(fieldVariant)) {
                VField_t<T, Dim>* field = std::get<VField_t<T, Dim>*>(fieldVariant);
                // == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>*

                Execute_Field(field, fieldName, vector_host_views[fieldName], node);     
            }
        }

        //AddSteerableChannel(node, scaleFactorE, scaleFactorB);

        // Pass Conduit node to Catalyst
        catalyst_status err = catalyst_execute(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to execute Catalyst: " << err << std::endl;
        }

        //Results(scaleFactorE, scaleFactorB);
    }

}  // namespace CatalystAdaptor

#endif
