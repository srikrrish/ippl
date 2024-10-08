//
// Class FFT
//   The FFT class performs complex-to-complex,
//   real-to-complex on IPPL Fields.
//   FFT is templated on the type of transform to be performed,
//   the dimensionality of the Field to transform, and the
//   floating-point precision type of the Field (float or double).
//   Currently, we use heffte for taking the transforms and the class FFT
//   serves as an interface between IPPL and heffte. In making this interface,
//   we have referred Cabana library.
//   https://github.com/ECP-copa/Cabana.
//
// Copyright (c) 2021, Sriramkrishnan Muralikrishnan,
// Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
/**
   Implementations for FFT constructor/destructor and transforms
*/

#include "FFT/FFT.h"
#include "FieldLayout/FieldLayout.h"
#include "Field/BareField.h"
#include "Utility/IpplTimings.h"


namespace ippl {

    //=========================================================================
    // FFT CCTransform Constructors
    //=========================================================================

    /**
       Create a new FFT object of type CCTransform, with a
       given layout and heffte parameters.
    */

    template <size_t Dim, class T>
    FFT<CCTransform,Dim,T>::FFT(
        const Layout_t& layout,
        const ParameterList& params)
    {


        /**
         * Heffte requires to pass a 3D array even for 2D and
         * 1D FFTs we just have to make the length in other
         * dimensions to be 1.
         */
        std::array<long long, 3> low;
        std::array<long long, 3> high;

        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();

        low.fill(0);
        high.fill(0);

        /**
         * Static cast to detail::long long (uint64_t) is necessary, as heffte::box3d requires it
         * like that.
         */
        for(size_t d = 0; d < Dim; ++d) {
            low[d] = static_cast<long long>(lDom[d].first());
            high[d] = static_cast<long long>(lDom[d].length() + lDom[d].first() - 1);
        }

        if(tempField_m.size() < lDom.size()) {
            Kokkos::realloc(tempField_m, lDom[0].length(), lDom[1].length(), lDom[2].length());
        }
        setup(low, high, params);
    }


    /**
           setup performs the initialization necessary.
    */
    template <size_t Dim, class T>
    void
    FFT<CCTransform,Dim,T>::setup(const std::array<long long, Dim>& low,
                                  const std::array<long long, Dim>& high,
                                  const ParameterList& params)
    {

        heffte::box3d<long long> inbox  = {low, high};
        heffte::box3d<long long> outbox = {low, high};

        heffte::plan_options heffteOptions =
            heffte::default_options<heffteBackend>();

        if(!params.get<bool>("use_heffte_defaults")) {
           heffteOptions.use_pencils = params.get<bool>("use_pencils");
           heffteOptions.use_reorder = params.get<bool>("use_reorder");
#ifdef Heffte_ENABLE_GPU
           heffteOptions.use_gpu_aware = params.get<bool>("use_gpu_aware");
#endif

           switch (params.get<int>("comm")) {
           
              case a2a:
                  heffteOptions.algorithm = heffte::reshape_algorithm::alltoall;
                  break;
              case a2av:
                  heffteOptions.algorithm = heffte::reshape_algorithm::alltoallv;
                  break;
              case p2p:
                  heffteOptions.algorithm = heffte::reshape_algorithm::p2p;
                  break;
              case p2p_pl:
                  heffteOptions.algorithm = heffte::reshape_algorithm::p2p_plined;
                  break;
              default:
                  throw IpplException("FFT::setup",
                                      "Unrecognized heffte communication type");
           }
        }

        heffte_m = std::make_shared<heffte::fft3d<heffteBackend, long long>>
                  (inbox, outbox, Ippl::getComm(), heffteOptions);

        //heffte::gpu::device_set(Ippl::Comm->rank() % heffte::gpu::device_count());
        if(workspace_m.size() < heffte_m->size_workspace())
           workspace_m = workspace_t(heffte_m->size_workspace());

    }



    template <size_t Dim, class T>
    void
    FFT<CCTransform,Dim,T>::transform(
        int direction,
        typename FFT<CCTransform,Dim,T>::ComplexField_t& f)
    {
        auto fview = f.getView();
        const int nghost = f.getNghost();

        /**
         *This copy to a temporary Kokkos view is needed because of following
         *reasons:
         *1) heffte wants the input and output fields without ghost layers
         *2) heffte accepts data in layout left (by default) eventhough this
         *can be changed during heffte box creation
        */
        //Kokkos::View<Complex_t***,Kokkos::LayoutLeft>
        //    tempField("tempField", fview.extent(0) - 2*nghost,
        //                           fview.extent(1) - 2*nghost,
        //                           fview.extent(2) - 2*nghost);

        auto tempField = tempField_m;
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        Kokkos::parallel_for("copy from Kokkos FFT",
                             mdrange_type({nghost, nghost, nghost},
                                          {fview.extent(0) - nghost,
                                           fview.extent(1) - nghost,
                                           fview.extent(2) - nghost
                                          }),
                             KOKKOS_LAMBDA(const size_t i,
                                           const size_t j,
                                           const size_t k)
                             {
                                 tempField(i-nghost, j-nghost, k-nghost).real(
                                       fview(i, j, k).real());
                                 tempField(i-nghost, j-nghost, k-nghost).imag(
                                       fview(i, j, k).imag());
                             });




        if ( direction == 1 )
        {
            heffte_m->forward(tempField.data(), tempField.data(), workspace_m.data(),
                              heffte::scale::full);
        }
        else if ( direction == -1 )
        {
            heffte_m->backward(tempField.data(), tempField.data(), workspace_m.data(),
                               heffte::scale::none);
        }
        else
        {
            throw std::logic_error(
                 "Only 1:forward and -1:backward are allowed as directions");
        }


        Kokkos::parallel_for("copy to Kokkos FFT",
                             mdrange_type({nghost, nghost, nghost},
                                          {fview.extent(0) - nghost,
                                           fview.extent(1) - nghost,
                                           fview.extent(2) - nghost
                                          }),
                             KOKKOS_LAMBDA(const size_t i,
                                           const size_t j,
                                           const size_t k)
                             {
                                 fview(i, j, k).real() =
                                 tempField(i-nghost, j-nghost, k-nghost).real();
                                 fview(i, j, k).imag() =
                                 tempField(i-nghost, j-nghost, k-nghost).imag();
                             });

    }


    //========================================================================
    // FFT RCTransform Constructors
    //========================================================================

    /**
       *Create a new FFT object of type RCTransform, with given input and output
       *layouts and heffte parameters.
    */

    template <size_t Dim, class T>
    FFT<RCTransform,Dim,T>::FFT(
        const Layout_t& layoutInput,
        const Layout_t& layoutOutput,
        const ParameterList& params)
    {

        /**
         * Heffte requires to pass a 3D array even for 2D and
         * 1D FFTs we just have to make the length in other
         * dimensions to be 1.
         */
        std::array<long long, 3> lowInput;
        std::array<long long, 3> highInput;
        std::array<long long, 3> lowOutput;
        std::array<long long, 3> highOutput;

        const NDIndex<Dim>& lDomInput = layoutInput.getLocalNDIndex();
        const NDIndex<Dim>& lDomOutput = layoutOutput.getLocalNDIndex();


        lowInput.fill(0);
        highInput.fill(0);
        lowOutput.fill(0);
        highOutput.fill(0);

        /**
         * Static cast to detail::long long (uint64_t) is necessary, as heffte::box3d requires it
         * like that.
         */
        for(size_t d = 0; d < Dim; ++d) {
            lowInput[d] = static_cast<long long>(lDomInput[d].first());
            highInput[d] = static_cast<long long>(lDomInput[d].length() +
                           lDomInput[d].first() - 1);

            lowOutput[d] = static_cast<long long>(lDomOutput[d].first());
            highOutput[d] = static_cast<long long>(lDomOutput[d].length() +
                            lDomOutput[d].first() - 1);
        }


        if(tempFieldf_m.size() < lDomInput.size()) {
            Kokkos::realloc(tempFieldf_m, lDomInput[0].length(), lDomInput[1].length(), lDomInput[2].length());
        }
        if(tempFieldg_m.size() < lDomOutput.size()) {
            Kokkos::realloc(tempFieldg_m, lDomOutput[0].length(), lDomOutput[1].length(), lDomOutput[2].length());
        }

        setup(lowInput, highInput, lowOutput, highOutput, params);
    }


    /**
       setup performs the initialization.
    */
    template <size_t Dim, class T>
    void
    FFT<RCTransform,Dim,T>::setup(const std::array<long long, Dim>& lowInput,
                                  const std::array<long long, Dim>& highInput,
                                  const std::array<long long, Dim>& lowOutput,
                                  const std::array<long long, Dim>& highOutput,
                                  const ParameterList& params)
    {

        heffte::box3d<long long> inbox  = {lowInput, highInput};
        heffte::box3d<long long> outbox = {lowOutput, highOutput};

        heffte::plan_options heffteOptions = 
            heffte::default_options<heffteBackend>();
        
        if(!params.get<bool>("use_heffte_defaults")) {
           heffteOptions.use_pencils = params.get<bool>("use_pencils");
           heffteOptions.use_reorder = params.get<bool>("use_reorder");
#ifdef Heffte_ENABLE_GPU
           heffteOptions.use_gpu_aware = params.get<bool>("use_gpu_aware");
#endif

            switch (params.get<int>("comm")) {
            
               case a2a:
                   heffteOptions.algorithm = heffte::reshape_algorithm::alltoall;
                   break;
               case a2av:
                   heffteOptions.algorithm = heffte::reshape_algorithm::alltoallv;
                   break;
               case p2p:
                   heffteOptions.algorithm = heffte::reshape_algorithm::p2p;
                   break;
               case p2p_pl:
                   heffteOptions.algorithm = heffte::reshape_algorithm::p2p_plined;
                   break;
               default:
                   throw IpplException("FFT::setup",
                                       "Unrecognized heffte communication type");
            }
         }

         heffte_m = std::make_shared<heffte::fft3d_r2c<heffteBackend, long long>>
                    (inbox, outbox, params.get<int>("r2c_direction"), MPI_COMM_SELF,
                     heffteOptions);
        
        //heffte::gpu::device_set(Ippl::Comm->rank() % heffte::gpu::device_count());
        if(workspace_m.size() < heffte_m->size_workspace())
           workspace_m = workspace_t(heffte_m->size_workspace());

    }

    template <size_t Dim, class T>
    void
    FFT<RCTransform,Dim,T>::transform(
        int direction,
        typename FFT<RCTransform,Dim,T>::RealField_t& f,
        typename FFT<RCTransform,Dim,T>::ComplexField_t& g)
    {
        auto fview = f.getView();
        auto gview = g.getView();
        const int nghostf = f.getNghost();
        const int nghostg = g.getNghost();

        /**
         *This copy to a temporary Kokkos view is needed because of following
         *reasons:
         *1) heffte wants the input and output fields without ghost layers
         *2) heffte accepts data in layout left (by default) eventhough this
         *can be changed during heffte box creation
        */
        //Kokkos::View<T***, Kokkos::LayoutLeft>
        //    tempFieldf("tempFieldf", fview.extent(0) - 2*nghostf,
        //                             fview.extent(1) - 2*nghostf,
        //                             fview.extent(2) - 2*nghostf);

        //Kokkos::View<Complex_t***, Kokkos::LayoutLeft>
        //    tempFieldg("tempFieldg", gview.extent(0) - 2*nghostg,
        //                             gview.extent(1) - 2*nghostg,
        //                             gview.extent(2) - 2*nghostg);

        auto tempFieldf = tempFieldf_m;
        auto tempFieldg = tempFieldg_m;
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        Kokkos::parallel_for("copy from Kokkos f field in FFT",
                             mdrange_type({nghostf, nghostf, nghostf},
                                          {fview.extent(0) - nghostf,
                                           fview.extent(1) - nghostf,
                                           fview.extent(2) - nghostf
                                          }),
                             KOKKOS_LAMBDA(const size_t i,
                                           const size_t j,
                                           const size_t k)
        {
             tempFieldf(i-nghostf, j-nghostf, k-nghostf) = fview(i, j, k);
        });
        Kokkos::parallel_for("copy from Kokkos g field in FFT",
                             mdrange_type({nghostg, nghostg, nghostg},
                                          {gview.extent(0) - nghostg,
                                           gview.extent(1) - nghostg,
                                           gview.extent(2) - nghostg
                                          }),
                             KOKKOS_LAMBDA(const size_t i,
                                           const size_t j,
                                           const size_t k)
                             {
                                 tempFieldg(i-nghostg, j-nghostg, k-nghostg).real(
                                       gview(i, j, k).real());
                                 tempFieldg(i-nghostg, j-nghostg, k-nghostg).imag(
                                       gview(i, j, k).imag());
                             });

      
        if ( direction == 1 )
        {
            heffte_m->forward( tempFieldf.data(), tempFieldg.data(), workspace_m.data(),
                               heffte::scale::full );
        }
        else if ( direction == -1 )
        {
            heffte_m->backward( tempFieldg.data(), tempFieldf.data(), workspace_m.data(),
                                heffte::scale::none );
        }
        else
        {
            throw std::logic_error(
                 "Only 1:forward and -1:backward are allowed as directions");
        }


        Kokkos::parallel_for("copy to Kokkos f field FFT",
                             mdrange_type({nghostf, nghostf, nghostf},
                                          {fview.extent(0) - nghostf,
                                           fview.extent(1) - nghostf,
                                           fview.extent(2) - nghostf
                                          }),
                             KOKKOS_LAMBDA(const size_t i,
                                           const size_t j,
                                           const size_t k)
        {
             fview(i, j, k) = tempFieldf(i-nghostf, j-nghostf, k-nghostf);
        });

        Kokkos::parallel_for("copy to Kokkos g field FFT",
                             mdrange_type({nghostg, nghostg, nghostg},
                                          {gview.extent(0) - nghostg,
                                           gview.extent(1) - nghostg,
                                           gview.extent(2) - nghostg
                                          }),
                             KOKKOS_LAMBDA(const size_t i,
                                           const size_t j,
                                           const size_t k)
                             {
                                 gview(i, j, k).real() =
                                 tempFieldg(i-nghostg, j-nghostg, k-nghostg).real();
                                 gview(i, j, k).imag() =
                                 tempFieldg(i-nghostg, j-nghostg, k-nghostg).imag();
                             });

    }

    //=========================================================================
    // FFT SineTransform Constructors
    //=========================================================================

    /**
       Create a new FFT object of type SineTransform, with a
       given layout and heffte parameters.
    */

    template <size_t Dim, class T>
    FFT<SineTransform,Dim,T>::FFT(
        const Layout_t& layout,
        const ParameterList& params)
    {

        /**
         * Heffte requires to pass a 3D array even for 2D and
         * 1D FFTs we just have to make the length in other
         * dimensions to be 1.
         */
        std::array<long long, 3> low;
        std::array<long long, 3> high;

        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();

        low.fill(0);
        high.fill(0);

        /**
         * Static cast to detail::long long (uint64_t) is necessary, as heffte::box3d requires it
         * like that.
         */
        for(size_t d = 0; d < Dim; ++d) {
            low[d] = static_cast<long long>(lDom[d].first());
            high[d] = static_cast<long long>(lDom[d].length() + lDom[d].first() - 1);
        }

        if(tempField_m.size() < lDom.size()) {
            Kokkos::realloc(tempField_m, lDom[0].length(), lDom[1].length(), lDom[2].length());
        }
        setup(low, high, params);
    }


    /**
           setup performs the initialization necessary.
    */
    template <size_t Dim, class T>
    void
    FFT<SineTransform,Dim,T>::setup(const std::array<long long, Dim>& low,
                                  const std::array<long long, Dim>& high,
                                  const ParameterList& params)
    {

        heffte::box3d<long long> inbox  = {low, high};
        heffte::box3d<long long> outbox = {low, high};

        heffte::plan_options heffteOptions =
            heffte::default_options<heffteBackend>();

        if(!params.get<bool>("use_heffte_defaults")) {
           heffteOptions.use_pencils = params.get<bool>("use_pencils");
           heffteOptions.use_reorder = params.get<bool>("use_reorder");
#ifdef Heffte_ENABLE_GPU
           heffteOptions.use_gpu_aware = params.get<bool>("use_gpu_aware");
#endif
           switch (params.get<int>("comm")) {
           
              case a2a:
                  heffteOptions.algorithm = heffte::reshape_algorithm::alltoall;
                  break;
              case a2av:
                  heffteOptions.algorithm = heffte::reshape_algorithm::alltoallv;
                  break;
              case p2p:
                  heffteOptions.algorithm = heffte::reshape_algorithm::p2p;
                  break;
              case p2p_pl:
                  heffteOptions.algorithm = heffte::reshape_algorithm::p2p_plined;
                  break;
              default:
                  throw IpplException("FFT::setup",
                                      "Unrecognized heffte communication type");
           }
        }

        heffte_m = std::make_shared<heffte::fft3d<heffteBackend, long long>>
                   (inbox, outbox, Ippl::getComm(), heffteOptions);

        //heffte::gpu::device_set(Ippl::Comm->rank() % heffte::gpu::device_count());
        if(workspace_m.size() < heffte_m->size_workspace())
           workspace_m = workspace_t(heffte_m->size_workspace());

    }

    template <size_t Dim, class T>
    void
    FFT<SineTransform,Dim,T>::transform(
        int direction,
        typename FFT<SineTransform,Dim,T>::Field_t& f)
    {
        auto fview = f.getView();
        const int nghost = f.getNghost();
        
        /**
         *This copy to a temporary Kokkos view is needed because of following
         *reasons:
         *1) heffte wants the input and output fields without ghost layers
         *2) heffte accepts data in layout left (by default) eventhough this
         *can be changed during heffte box creation
        */
        //Kokkos::View<T***,Kokkos::LayoutLeft>
        //    tempField("tempField", fview.extent(0) - 2*nghost,
        //                           fview.extent(1) - 2*nghost,
        //                           fview.extent(2) - 2*nghost);

        auto tempField = tempField_m;
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        Kokkos::parallel_for("copy from Kokkos FFT",
                             mdrange_type({nghost, nghost, nghost},
                                          {fview.extent(0) - nghost,
                                           fview.extent(1) - nghost,
                                           fview.extent(2) - nghost
                                          }),
                             KOKKOS_LAMBDA(const size_t i,
                                           const size_t j,
                                           const size_t k)
                             {
                               tempField(i-nghost, j-nghost, k-nghost) = 
                               fview(i, j, k);
                             });

        if ( direction == 1 )
        {
            heffte_m->forward(tempField.data(), tempField.data(), workspace_m.data(),
                              heffte::scale::full);
        }
        else if ( direction == -1 )
        {
            heffte_m->backward(tempField.data(), tempField.data(), workspace_m.data(),
                               heffte::scale::none);
        }
        else
        {
            throw std::logic_error(
                 "Only 1:forward and -1:backward are allowed as directions");
        }

        Kokkos::parallel_for("copy to Kokkos FFT",
                             mdrange_type({nghost, nghost, nghost},
                                          {fview.extent(0) - nghost,
                                           fview.extent(1) - nghost,
                                           fview.extent(2) - nghost
                                          }),
                             KOKKOS_LAMBDA(const size_t i,
                                           const size_t j,
                                           const size_t k)
                             {
                               fview(i, j, k) =
                               tempField(i-nghost, j-nghost, k-nghost);
                             });

    }

    //=========================================================================
    // FFT CosTransform Constructors
    //=========================================================================

    /**
       Create a new FFT object of type CosTransform, with a
       given layout and heffte parameters.
    */

    template <size_t Dim, class T>
    FFT<CosTransform,Dim,T>::FFT(
        const Layout_t& layout,
        const ParameterList& params)
    {

        /**
         * Heffte requires to pass a 3D array even for 2D and
         * 1D FFTs we just have to make the length in other
         * dimensions to be 1.
         */
        std::array<long long, 3> low;
        std::array<long long, 3> high;

        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();

        low.fill(0);
        high.fill(0);

        /**
         * Static cast to detail::long long (uint64_t) is necessary, as heffte::box3d requires it
         * like that.
         */
        for(size_t d = 0; d < Dim; ++d) {
            low[d] = static_cast<long long>(lDom[d].first());
            high[d] = static_cast<long long>(lDom[d].length() + lDom[d].first() - 1);
        }

        if(tempField_m.size() < lDom.size()) {
            Kokkos::realloc(tempField_m, lDom[0].length(), lDom[1].length(), lDom[2].length());
        }
        setup(low, high, params);
    }


    /**
           setup performs the initialization necessary.
    */
    template <size_t Dim, class T>
    void
    FFT<CosTransform,Dim,T>::setup(const std::array<long long, Dim>& low,
                                  const std::array<long long, Dim>& high,
                                  const ParameterList& params)
    {

        heffte::box3d<long long> inbox  = {low, high};
        heffte::box3d<long long> outbox = {low, high};

        heffte::plan_options heffteOptions =
            heffte::default_options<heffteBackend>();

        if(!params.get<bool>("use_heffte_defaults")) {
           heffteOptions.use_pencils = params.get<bool>("use_pencils");
           heffteOptions.use_reorder = params.get<bool>("use_reorder");
#ifdef Heffte_ENABLE_GPU
           heffteOptions.use_gpu_aware = params.get<bool>("use_gpu_aware");
#endif
           switch (params.get<int>("comm")) {
           
              case a2a:
                  heffteOptions.algorithm = heffte::reshape_algorithm::alltoall;
                  break;
              case a2av:
                  heffteOptions.algorithm = heffte::reshape_algorithm::alltoallv;
                  break;
              case p2p:
                  heffteOptions.algorithm = heffte::reshape_algorithm::p2p;
                  break;
              case p2p_pl:
                  heffteOptions.algorithm = heffte::reshape_algorithm::p2p_plined;
                  break;
              default:
                  throw IpplException("FFT::setup",
                                      "Unrecognized heffte communication type");
           }
        }

        heffte_m = std::make_shared<heffte::fft3d<heffteBackend, long long>>
                   (inbox, outbox, Ippl::getComm(), heffteOptions);

        //heffte::gpu::device_set(Ippl::Comm->rank() % heffte::gpu::device_count());
        if(workspace_m.size() < heffte_m->size_workspace())
           workspace_m = workspace_t(heffte_m->size_workspace());

    }


    template <size_t Dim, class T>
    void
    FFT<CosTransform,Dim,T>::transform(
        int direction,
        typename FFT<CosTransform,Dim,T>::Field_t& f)
    {
        auto fview = f.getView();
        const int nghost = f.getNghost();

        /**
         *This copy to a temporary Kokkos view is needed because of following
         *reasons:
         *1) heffte wants the input and output fields without ghost layers
         *2) heffte accepts data in layout left (by default) eventhough this
         *can be changed during heffte box creation
        */
        //Kokkos::View<T***,Kokkos::LayoutLeft>
        //    tempField("tempField", fview.extent(0) - 2*nghost,
        //                           fview.extent(1) - 2*nghost,
        //                           fview.extent(2) - 2*nghost);

        auto tempField = tempField_m;
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        Kokkos::parallel_for("copy from Kokkos FFT",
                             mdrange_type({nghost, nghost, nghost},
                                          {fview.extent(0) - nghost,
                                           fview.extent(1) - nghost,
                                           fview.extent(2) - nghost
                                          }),
                             KOKKOS_LAMBDA(const size_t i,
                                           const size_t j,
                                           const size_t k)
                             {
                               tempField(i-nghost, j-nghost, k-nghost) = 
                               fview(i, j, k);
                             });

        if ( direction == 1 )
        {
            heffte_m->forward(tempField.data(), tempField.data(), workspace_m.data(),
                              heffte::scale::full);
        }
        else if ( direction == -1 )
        {
            heffte_m->backward(tempField.data(), tempField.data(), workspace_m.data(),
                               heffte::scale::none);
        }
        else
        {
            throw std::logic_error(
                 "Only 1:forward and -1:backward are allowed as directions");
        }

        Kokkos::parallel_for("copy to Kokkos FFT",
                             mdrange_type({nghost, nghost, nghost},
                                          {fview.extent(0) - nghost,
                                           fview.extent(1) - nghost,
                                           fview.extent(2) - nghost
                                          }),
                             KOKKOS_LAMBDA(const size_t i,
                                           const size_t j,
                                           const size_t k)
                             {
                               fview(i, j, k) =
                               tempField(i-nghost, j-nghost, k-nghost);
                             });

    }


#ifdef KOKKOS_ENABLE_CUDA
    //=========================================================================
    // FFT NUFFTransform Constructors
    //=========================================================================

    /**
       Create a new FFT object of type NUFFTransform, with a
       given layout and cuFINUFFT parameters.
    */

    template <size_t Dim, class T>
    FFT<NUFFTransform,Dim,T>::FFT(const Layout_t& layout,
                                  const detail::size_type& localNp,
                                  int type,
                                  const ParameterList& params)
    {
        /**
         * cuFINUFFT requires to pass a 3D array even for 2D and
         * 1D FFTs we just have to fill in other
         * dimensions to be 1. Note this is different from Heffte
         * where we fill 0.
         */
        
        std::array<int64_t, 3> nmodes;

        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();

        nmodes.fill(1);
    
        for(size_t d = 0; d < Dim; ++d) {
            nmodes[d] = lDom[d].length();;
        }

        type_m = type;
        if(tempField_m.size() < lDom.size()) {
            Kokkos::realloc(tempField_m, lDom[0].length(), lDom[1].length(), lDom[2].length());
        }
        for(size_t d = 0; d < Dim; ++d) {
            if(tempR_m[d].size() < localNp) {
                Kokkos::realloc(tempR_m[d], localNp);
            }
        }
        if(tempQ_m.size() < localNp) {
            Kokkos::realloc(tempQ_m, localNp);
        }
        setup(nmodes, params);
    }


    /**
        setup performs the initialization necessary.
    */
    template <size_t Dim, class T>
    void
    FFT<NUFFTransform,Dim,T>::setup(std::array<int64_t, 3>& nmodes,
                                    const ParameterList& params)
    {

        cufinufft_opts opts;
	    cufinufft_default_opts(&opts);
        tol_m = 1e-6;

        if(!params.get<bool>("use_cufinufft_defaults")) {
           tol_m = params.get<T>("tolerance");
           opts.gpu_method = params.get<int>("gpu_method");
           opts.gpu_sort = params.get<int>("gpu_sort");
           opts.gpu_kerevalmeth = params.get<int>("gpu_kerevalmeth");
        }

	    opts.gpu_maxbatchsize = 0; //default option. ignored for ntransf = 1 which
                                   // is our case
	    //For Perlmutter since the mask to hide the other GPUs in the node is 
        //somehow not working there
        //opts.gpu_device_id = (int)(Ippl::Comm->rank() % 4);

        int iflag;
        
        if(type_m == 1) {
            iflag = -1;
        }
        else if(type_m == 2) {
            iflag = 1;
        }
        else {
            throw std::logic_error("Only type 1 and type 2 NUFFT are allowed now");
        }

        //dim in cufinufft is int
        int dim = static_cast<int>(Dim);
        ier_m = nufft_m.makeplan(type_m, dim, nmodes.data(), iflag, 1, tol_m,
                       		 &plan_m, &opts);  

    }



    template <size_t Dim, class T>
    template<class... Properties>
    void
    FFT<NUFFTransform,Dim,T>::transform(const ParticleAttrib< Vector<T, Dim>, Properties... >& R,
                                        ParticleAttrib<T, Properties... >& Q,
                                        typename FFT<NUFFTransform,Dim,T>::ComplexField_t& f)
    {
        
        
        auto fview = f.getView();
        auto Rview = R.getView();
        auto Qview = Q.getView();
        const int nghost = f.getNghost();

        auto localNp = R.getParticleCount();

        const Layout_t& layout = f.getLayout(); 
        const UniformCartesian<T, Dim>& mesh = f.get_mesh();
        const Vector<T, Dim>& dx = mesh.getMeshSpacing();
        const Vector<T, Dim>& origin = mesh.getOrigin();
        const auto& domain = layout.getDomain();
        Vector<T, Dim> Len;
        Vector<int, Dim> N;

        for (unsigned d=0; d < Dim; ++d) {
            N[d] = domain[d].length();
            Len[d] = dx[d] * N[d];
        }

        const double pi = std::acos(-1.0);

        auto tempField = tempField_m;
        auto tempQ = tempQ_m;
        Kokkos::View<T*,Kokkos::LayoutLeft> tempR[3] = {};
        for(size_t d = 0; d < Dim; ++d) {
            tempR[d] = tempR_m[d];
        }
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        Kokkos::parallel_for("copy from field data NUFFT",
                             mdrange_type({nghost, nghost, nghost},
                                          {fview.extent(0) - nghost,
                                           fview.extent(1) - nghost,
                                           fview.extent(2) - nghost
                                          }),
                             KOKKOS_LAMBDA(const size_t i,
                                           const size_t j,
                                           const size_t k)
                             {
                                 tempField(i-nghost, j-nghost, k-nghost).x =
                                       fview(i, j, k).real();
                                 tempField(i-nghost, j-nghost, k-nghost).y = 
                                       fview(i, j, k).imag();
                             });


        Kokkos::parallel_for("copy from particle data NUFFT",
                             localNp,
                             KOKKOS_LAMBDA(const size_t i)
                             {
                                 for(size_t d = 0; d < Dim; ++d) {
                                    //tempR[d](i) = (Rview(i)[d] - (twopiFactor * 2.0 * pi)) * (2.0 * pi / Len[d]);
                                    tempR[d](i) = Rview(i)[d] * (2.0 * pi / Len[d]);
                                    //tempR[d](i) = Rview(i)[d];
                                 }
                                 tempQ(i).x = Qview(i);
                                 tempQ(i).y = 0.0;
                             });

        ier_m = nufft_m.setpts(plan_m, localNp, tempR[0].data(), tempR[1].data(), tempR[2].data(), 0, 
                     NULL, NULL, NULL);

        ier_m = nufft_m.execute(plan_m, tempQ.data(), tempField.data());
        Kokkos::fence();


        if(type_m == 1) { 
            Kokkos::parallel_for("copy to field data NUFFT",
                                 mdrange_type({nghost, nghost, nghost},
                                              {fview.extent(0) - nghost,
                                               fview.extent(1) - nghost,
                                               fview.extent(2) - nghost
                                              }),
                                 KOKKOS_LAMBDA(const size_t i,
                                               const size_t j,
                                               const size_t k)
                                 {
                                     fview(i, j, k).real() =
                                     tempField(i-nghost, j-nghost, k-nghost).x;
                                     fview(i, j, k).imag() =
                                     tempField(i-nghost, j-nghost, k-nghost).y;
                                 });
        }
        else if(type_m == 2) {
            Kokkos::parallel_for("copy to particle data NUFFT",
                                 localNp,
                                 KOKKOS_LAMBDA(const size_t i)
                                 {
                                     Qview(i) = tempQ(i).x;
                                 });
        }
    }
    
    template <size_t Dim, class T>
    FFT<NUFFTransform,Dim,T>::~FFT() {
        
        ier_m = nufft_m.destroy(plan_m);

    }
#endif
}

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
