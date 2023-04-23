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

#include "Utility/IpplTimings.h"

#include "Field/BareField.h"

#include "FFT/FFT.h"
#include "FieldLayout/FieldLayout.h"

namespace ippl {

    //=========================================================================
    // FFT CCTransform Constructors
    //=========================================================================

    /**
       Create a new FFT object of type CCTransform, with a
       given layout and heffte parameters.
    */

    template <size_t Dim, class T, class Mesh, class Centering>
    FFT<CCTransform, Dim, T, Mesh, Centering>::FFT(const Layout_t& layout,
                                                   const ParameterList& params) {
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
        for (size_t d = 0; d < Dim; ++d) {
            low[d]  = static_cast<long long>(lDom[d].first());
            high[d] = static_cast<long long>(lDom[d].length() + lDom[d].first() - 1);
        }

        setup(low, high, params);
    }

    /**
           setup performs the initialization necessary.
    */
    template <size_t Dim, class T, class Mesh, class Centering>
    void FFT<CCTransform, Dim, T, Mesh, Centering>::setup(const std::array<long long, 3>& low,
                                                          const std::array<long long, 3>& high,
                                                          const ParameterList& params) {
        heffte::box3d<long long> inbox  = {low, high};
        heffte::box3d<long long> outbox = {low, high};

        heffte::plan_options heffteOptions = heffte::default_options<heffteBackend>();

        if (!params.get<bool>("use_heffte_defaults")) {
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
                    throw IpplException("FFT::setup", "Unrecognized heffte communication type");
            }
        }

        heffte_m = std::make_shared<heffte::fft3d<heffteBackend, long long>>(
            inbox, outbox, Ippl::getComm(), heffteOptions);

        // heffte::gpu::device_set(Ippl::Comm->rank() % heffte::gpu::device_count());
        if (workspace_m.size() < heffte_m->size_workspace())
            workspace_m = workspace_t(heffte_m->size_workspace());
    }

    template <size_t Dim, class T, class Mesh, class Centering>
    void FFT<CCTransform, Dim, T, Mesh, Centering>::transform(
        int direction, typename FFT<CCTransform, Dim, T, Mesh, Centering>::ComplexField_t& f) {
        static_assert(Dim <= 3, "heFFTe doesn't support Dim > 3 yet");

        auto fview       = f.getView();
        const int nghost = f.getNghost();

        /**
         *This copy to a temporary Kokkos view is needed because of following
         *reasons:
         *1) heffte wants the input and output fields without ghost layers
         *2) heffte accepts data in layout left (by default) eventhough this
         *can be changed during heffte box creation
         */
        auto tempField = detail::shrinkView<Dim, Complex_t>("tempField", fview, nghost);

        using index_array_type = typename detail::RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "copy from Kokkos FFT", detail::getRangePolicy<Dim>(fview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply<Dim>(tempField, args - nghost).real(apply<Dim>(fview, args).real());
                apply<Dim>(tempField, args - nghost).imag(apply<Dim>(fview, args).imag());
            });

        if (direction == 1) {
            heffte_m->forward(tempField.data(), tempField.data(), workspace_m.data(),
                              heffte::scale::full);
        } else if (direction == -1) {
            heffte_m->backward(tempField.data(), tempField.data(), workspace_m.data(),
                               heffte::scale::none);
        } else {
            throw std::logic_error("Only 1:forward and -1:backward are allowed as directions");
        }

        ippl::parallel_for(
            "copy to Kokkos FFT", detail::getRangePolicy<Dim>(fview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply<Dim>(fview, args).real() = apply<Dim>(tempField, args - nghost).real();
                apply<Dim>(fview, args).imag() = apply<Dim>(tempField, args - nghost).imag();
            });
    }

    //========================================================================
    // FFT RCTransform Constructors
    //========================================================================

    /**
     *Create a new FFT object of type RCTransform, with given input and output
     *layouts and heffte parameters.
     */

    template <size_t Dim, class T, class Mesh, class Centering>
    FFT<RCTransform, Dim, T, Mesh, Centering>::FFT(const Layout_t& layoutInput,
                                                   const Layout_t& layoutOutput,
                                                   const ParameterList& params) {
        /**
         * Heffte requires to pass a 3D array even for 2D and
         * 1D FFTs we just have to make the length in other
         * dimensions to be 1.
         */
        std::array<long long, 3> lowInput;
        std::array<long long, 3> highInput;
        std::array<long long, 3> lowOutput;
        std::array<long long, 3> highOutput;

        const NDIndex<Dim>& lDomInput  = layoutInput.getLocalNDIndex();
        const NDIndex<Dim>& lDomOutput = layoutOutput.getLocalNDIndex();

        lowInput.fill(0);
        highInput.fill(0);
        lowOutput.fill(0);
        highOutput.fill(0);

        /**
         * Static cast to detail::long long (uint64_t) is necessary, as heffte::box3d requires it
         * like that.
         */
        for (size_t d = 0; d < Dim; ++d) {
            lowInput[d]  = static_cast<long long>(lDomInput[d].first());
            highInput[d] = static_cast<long long>(lDomInput[d].length() + lDomInput[d].first() - 1);

            lowOutput[d] = static_cast<long long>(lDomOutput[d].first());
            highOutput[d] =
                static_cast<long long>(lDomOutput[d].length() + lDomOutput[d].first() - 1);
        }

        setup(lowInput, highInput, lowOutput, highOutput, params);
    }

    /**
       setup performs the initialization.
    */
    template <size_t Dim, class T, class Mesh, class Centering>
    void FFT<RCTransform, Dim, T, Mesh, Centering>::setup(
        const std::array<long long, 3>& lowInput, const std::array<long long, 3>& highInput,
        const std::array<long long, 3>& lowOutput, const std::array<long long, 3>& highOutput,
        const ParameterList& params) {
        heffte::box3d<long long> inbox  = {lowInput, highInput};
        heffte::box3d<long long> outbox = {lowOutput, highOutput};

        heffte::plan_options heffteOptions = heffte::default_options<heffteBackend>();

        if (!params.get<bool>("use_heffte_defaults")) {
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
                    throw IpplException("FFT::setup", "Unrecognized heffte communication type");
            }
        }

        heffte_m = std::make_shared<heffte::fft3d_r2c<heffteBackend, long long>>(
            inbox, outbox, params.get<int>("r2c_direction"), Ippl::getComm(), heffteOptions);

        // heffte::gpu::device_set(Ippl::Comm->rank() % heffte::gpu::device_count());
        if (workspace_m.size() < heffte_m->size_workspace())
            workspace_m = workspace_t(heffte_m->size_workspace());
    }

    template <size_t Dim, class T, class Mesh, class Centering>
    void FFT<RCTransform, Dim, T, Mesh, Centering>::transform(
        int direction, typename FFT<RCTransform, Dim, T, Mesh, Centering>::RealField_t& f,
        typename FFT<RCTransform, Dim, T, Mesh, Centering>::ComplexField_t& g) {
        static_assert(Dim <= 3, "heFFTe doesn't support Dim > 3 yet");

        auto fview        = f.getView();
        auto gview        = g.getView();
        const int nghostf = f.getNghost();
        const int nghostg = g.getNghost();

        /**
         *This copy to a temporary Kokkos view is needed because of following
         *reasons:
         *1) heffte wants the input and output fields without ghost layers
         *2) heffte accepts data in layout left (by default) eventhough this
         *can be changed during heffte box creation
         */
        auto tempFieldf = detail::shrinkView<Dim, T>("tempFieldf", fview, nghostf);
        auto tempFieldg = detail::shrinkView<Dim, Complex_t>("tempFieldg", gview, nghostg);

        using index_array_type = typename detail::RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "copy from Kokkos f field in FFT", detail::getRangePolicy<Dim>(fview, nghostf),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply<Dim>(tempFieldf, args - nghostf) = apply<Dim>(fview, args);
            });
        ippl::parallel_for(
            "copy from Kokkos g field in FFT", detail::getRangePolicy<Dim>(gview, nghostg),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply<Dim>(tempFieldg, args - nghostg).real(apply<Dim>(gview, args).real());
                apply<Dim>(tempFieldg, args - nghostg).imag(apply<Dim>(gview, args).imag());
            });

        if (direction == 1) {
            heffte_m->forward(tempFieldf.data(), tempFieldg.data(), workspace_m.data(),
                              heffte::scale::full);
        } else if (direction == -1) {
            heffte_m->backward(tempFieldg.data(), tempFieldf.data(), workspace_m.data(),
                               heffte::scale::none);
        } else {
            throw std::logic_error("Only 1:forward and -1:backward are allowed as directions");
        }

        ippl::parallel_for(
            "copy to Kokkos f field FFT", detail::getRangePolicy<Dim>(fview, nghostf),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply<Dim>(fview, args) = apply<Dim>(tempFieldf, args - nghostf);
            });

        ippl::parallel_for(
            "copy to Kokkos g field FFT", detail::getRangePolicy<Dim>(gview, nghostg),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply<Dim>(gview, args).real() = apply<Dim>(tempFieldg, args - nghostg).real();
                apply<Dim>(gview, args).imag() = apply<Dim>(tempFieldg, args - nghostg).imag();
            });
    }

    //=========================================================================
    // FFT SineTransform Constructors
    //=========================================================================

    /**
       Create a new FFT object of type SineTransform, with a
       given layout and heffte parameters.
    */

    template <size_t Dim, class T, class Mesh, class Centering>
    FFT<SineTransform, Dim, T, Mesh, Centering>::FFT(const Layout_t& layout,
                                                     const ParameterList& params) {
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
        for (size_t d = 0; d < Dim; ++d) {
            low[d]  = static_cast<long long>(lDom[d].first());
            high[d] = static_cast<long long>(lDom[d].length() + lDom[d].first() - 1);
        }

        setup(low, high, params);
    }

    /**
           setup performs the initialization necessary.
    */
    template <size_t Dim, class T, class Mesh, class Centering>
    void FFT<SineTransform, Dim, T, Mesh, Centering>::setup(const std::array<long long, 3>& low,
                                                            const std::array<long long, 3>& high,
                                                            const ParameterList& params) {
        heffte::box3d<long long> inbox  = {low, high};
        heffte::box3d<long long> outbox = {low, high};

        heffte::plan_options heffteOptions = heffte::default_options<heffteBackend>();

        if (!params.get<bool>("use_heffte_defaults")) {
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
                    throw IpplException("FFT::setup", "Unrecognized heffte communication type");
            }
        }

        heffte_m = std::make_shared<heffte::fft3d<heffteBackend, long long>>(
            inbox, outbox, Ippl::getComm(), heffteOptions);

        // heffte::gpu::device_set(Ippl::Comm->rank() % heffte::gpu::device_count());
        if (workspace_m.size() < heffte_m->size_workspace())
            workspace_m = workspace_t(heffte_m->size_workspace());
    }

    template <size_t Dim, class T, class Mesh, class Centering>
    void FFT<SineTransform, Dim, T, Mesh, Centering>::transform(
        int direction, typename FFT<SineTransform, Dim, T, Mesh, Centering>::Field_t& f) {
        static_assert(Dim <= 3, "heFFTe doesn't support Dim > 3 yet");

        auto fview       = f.getView();
        const int nghost = f.getNghost();

        /**
         *This copy to a temporary Kokkos view is needed because of following
         *reasons:
         *1) heffte wants the input and output fields without ghost layers
         *2) heffte accepts data in layout left (by default) eventhough this
         *can be changed during heffte box creation
         */
        auto tempField = detail::shrinkView<Dim, T>("tempField", fview, nghost);

        using index_array_type = typename detail::RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "copy from Kokkos FFT", detail::getRangePolicy<Dim>(fview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply<Dim>(tempField, args - nghost) = apply<Dim>(fview, args);
            });

        if (direction == 1) {
            heffte_m->forward(tempField.data(), tempField.data(), workspace_m.data(),
                              heffte::scale::full);
        } else if (direction == -1) {
            heffte_m->backward(tempField.data(), tempField.data(), workspace_m.data(),
                               heffte::scale::none);
        } else {
            throw std::logic_error("Only 1:forward and -1:backward are allowed as directions");
        }

        ippl::parallel_for(
            "copy to Kokkos FFT", detail::getRangePolicy<Dim>(fview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply<Dim>(fview, args) = apply<Dim>(tempField, args - nghost);
            });
    }

    //=========================================================================
    // FFT CosTransform Constructors
    //=========================================================================

    /**
       Create a new FFT object of type CosTransform, with a
       given layout and heffte parameters.
    */

    template <size_t Dim, class T, class Mesh, class Centering>
    FFT<CosTransform, Dim, T, Mesh, Centering>::FFT(const Layout_t& layout,
                                                    const ParameterList& params) {
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
        for (size_t d = 0; d < Dim; ++d) {
            low[d]  = static_cast<long long>(lDom[d].first());
            high[d] = static_cast<long long>(lDom[d].length() + lDom[d].first() - 1);
        }

        setup(low, high, params);
    }

    /**
           setup performs the initialization necessary.
    */
    template <size_t Dim, class T, class Mesh, class Centering>
    void FFT<CosTransform, Dim, T, Mesh, Centering>::setup(const std::array<long long, 3>& low,
                                                           const std::array<long long, 3>& high,
                                                           const ParameterList& params) {
        heffte::box3d<long long> inbox  = {low, high};
        heffte::box3d<long long> outbox = {low, high};

        heffte::plan_options heffteOptions = heffte::default_options<heffteBackend>();

        if (!params.get<bool>("use_heffte_defaults")) {
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
                    throw IpplException("FFT::setup", "Unrecognized heffte communication type");
            }
        }

        heffte_m = std::make_shared<heffte::fft3d<heffteBackend, long long>>(
            inbox, outbox, Ippl::getComm(), heffteOptions);

        // heffte::gpu::device_set(Ippl::Comm->rank() % heffte::gpu::device_count());
        if (workspace_m.size() < heffte_m->size_workspace())
            workspace_m = workspace_t(heffte_m->size_workspace());
    }

    template <size_t Dim, class T, class Mesh, class Centering>
    void FFT<CosTransform, Dim, T, Mesh, Centering>::transform(
        int direction, typename FFT<CosTransform, Dim, T, Mesh, Centering>::Field_t& f) {
        static_assert(Dim <= 3, "heFFTe doesn't support Dim > 3 yet");

        auto fview       = f.getView();
        const int nghost = f.getNghost();

        /**
         *This copy to a temporary Kokkos view is needed because of following
         *reasons:
         *1) heffte wants the input and output fields without ghost layers
         *2) heffte accepts data in layout left (by default) eventhough this
         *can be changed during heffte box creation
         */
        auto tempField = detail::shrinkView<Dim, T>("tempField", fview, nghost);

        using index_array_type = typename detail::RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "copy from Kokkos FFT", detail::getRangePolicy<Dim>(fview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply<Dim>(tempField, args - nghost) = apply<Dim>(fview, args);
            });

        if (direction == 1) {
            heffte_m->forward(tempField.data(), tempField.data(), workspace_m.data(),
                              heffte::scale::full);
        } else if (direction == -1) {
            heffte_m->backward(tempField.data(), tempField.data(), workspace_m.data(),
                               heffte::scale::none);
        } else {
            throw std::logic_error("Only 1:forward and -1:backward are allowed as directions");
        }

        ippl::parallel_for(
            "copy to Kokkos FFT", detail::getRangePolicy<Dim>(fview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply<Dim>(fview, args) = apply<Dim>(tempField, args - nghost);
            });
    }
}  // namespace ippl

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
