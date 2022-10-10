#ifndef CUDADataFormats_Track_TrajectoryStateSOAT_H
#define CUDADataFormats_Track_TrajectoryStateSOAT_H

#include "HeterogeneousCore/CUDAUtilities/interface/eigenSoA.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
using Vector5f = Eigen::Matrix<float, 5, 1>;
using Vector15f = Eigen::Matrix<float, 15, 1>;

using Vector5d = Eigen::Matrix<double, 5, 1>;
using Matrix5d = Eigen::Matrix<double, 5, 5>;
GENERATE_SOA_LAYOUT(TrajectoryStateSoAT_test,
                    SOA_EIGEN_COLUMN(Vector5f, state),
                    SOA_EIGEN_COLUMN(Vector15f, covariance))

template <int32_t S>
struct TrajectoryStateSoAT : public cms::cuda::PortableDeviceCollection<TrajectoryStateSoAT_test<>> {
  static constexpr int32_t stride() { return S; }

  // eigenSoA::MatrixSoA<Vector5f, S> state;
  // eigenSoA::MatrixSoA<Vector15f, S> covariance;

  // Vector5f state(const int32_t i) const { return view()[i].state(); }
  // float* state() const { return view().state(); }  // TODO: Return Vector5f* ?
  // Vector15f covariance(const int32_t i) const { return view()[i].covariance(); }
  // float* covariance() const { return view().covariance(); }  // TODO: Return Vector15f* ?

  // Restrict view
  // using RestrictConstView =
  //     Layout::ConstViewTemplate<cms::soa::RestrictQualify::enabled, cms::soa::RangeChecking::disabled>;

  // RestrictConstView restrictConstView() const { return RestrictConstView(layout()); }

  template <typename V3, typename M3, typename V2, typename M2>
  __host__ __device__ inline void copyFromCircle(
      V3 const& cp, M3 const& ccov, V2 const& lp, M2 const& lcov, float b, int32_t i) {
    view()[i].state() << cp.template cast<float>(), lp.template cast<float>();
    view()[i].state()(2) *= b;
    auto cov = view()[i].covariance();
    cov(0) = ccov(0, 0);
    cov(1) = ccov(0, 1);
    cov(2) = b * float(ccov(0, 2));
    cov(4) = cov(3) = 0;
    cov(5) = ccov(1, 1);
    cov(6) = b * float(ccov(1, 2));
    cov(8) = cov(7) = 0;
    cov(9) = b * b * float(ccov(2, 2));
    cov(11) = cov(10) = 0;
    cov(12) = lcov(0, 0);
    cov(13) = lcov(0, 1);
    cov(14) = lcov(1, 1);
  }

  template <typename V5, typename M5>
  __host__ __device__ inline void copyFromDense(V5 const& v, M5 const& cov, int32_t i) {
    view()[i].state() = v.template cast<float>();
    for (int j = 0, ind = 0; j < 5; ++j)
      for (auto k = j; k < 5; ++k)
        view()[i].covariance()(ind++) = cov(j, k);
  }

  template <typename V5, typename M5>
  __host__ __device__ inline void copyToDense(V5& v, M5& cov, int32_t i) const {
    v = view()[i].state().template cast<typename V5::Scalar>();
    for (int j = 0, ind = 0; j < 5; ++j) {
      cov(j, j) = view()[i].covariance()(ind++);
      for (auto k = j + 1; k < 5; ++k)
        cov(k, j) = cov(j, k) = view()[i].covariance()(ind++);
    }
  }
};

#endif  // CUDADataFormats_Track_TrajectoryStateSOAT_H
