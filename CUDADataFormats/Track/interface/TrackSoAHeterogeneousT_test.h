#ifndef CUDADataFormats_Track_TrackHeterogeneousT_H
#define CUDADataFormats_Track_TrackHeterogeneousT_H

#include <string>
#include <algorithm>

#include <Eigen/Dense>
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"

namespace pixelTrack {
  enum class Quality : uint8_t { bad = 0, edup, dup, loose, strict, tight, highPurity, notQuality };
  constexpr uint32_t qualitySize{uint8_t(Quality::notQuality)};
  const std::string qualityName[qualitySize]{"bad", "edup", "dup", "loose", "strict", "tight", "highPurity"};
  inline Quality qualityByName(std::string const &name) {
    auto qp = std::find(qualityName, qualityName + qualitySize, name) - qualityName;
    return static_cast<Quality>(qp);
  }
}  // namespace pixelTrack

using Vector5f = Eigen::Matrix<float, 5, 1>;
using Vector15f = Eigen::Matrix<float, 15, 1>;

using Vector5d = Eigen::Matrix<double, 5, 1>;
using Matrix5d = Eigen::Matrix<double, 5, 5>;

GENERATE_SOA_LAYOUT(TrackSoAHeterogeneousT_test,
                    SOA_COLUMN(uint8_t, quality),
                    SOA_COLUMN(float, chi2),  // this is chi2/ndof as not necessarely all hits are used in the fit
                    SOA_COLUMN(int8_t, nLayers),
                    SOA_COLUMN(float, eta),
                    SOA_COLUMN(float, pt),
                    SOA_EIGEN_COLUMN(Vector5f, state),
                    SOA_EIGEN_COLUMN(Vector15f, covariance),
                    SOA_SCALAR(int, nTracks))

// Previous TrajectoryStateSoAT class methods
namespace pixelTrack {
  namespace utilities {
    using TrackSoAView = cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::View;
    using TrackSoAConstView = cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::ConstView;
    // State at the Beam spot
    // phi,tip,1/pt,cotan(theta),zip
    __host__ __device__ inline float charge(TrackSoAConstView tracks, int32_t i) {
      return std::copysign(1.f, tracks[i].state()(2));
    }

    __host__ __device__ inline float phi(TrackSoAConstView tracks, int32_t i) { return tracks[i].state()(0); }

    __host__ __device__ inline float tip(TrackSoAConstView tracks, int32_t i) { return tracks[i].state()(1); }

    __host__ __device__ inline float zip(TrackSoAConstView tracks, int32_t i) { return tracks[i].state()(4); }

    __host__ __device__ inline bool isTriplet(TrackSoAConstView tracks, int i) { return tracks[i].nLayers() == 3; }

    template <typename V3, typename M3, typename V2, typename M2>
    __host__ __device__ inline void copyFromCircle(
        TrackSoAView tracks, V3 const &cp, M3 const &ccov, V2 const &lp, M2 const &lcov, float b, int32_t i) {
      tracks[i].state() << cp.template cast<float>(), lp.template cast<float>();

      tracks[i].state()(2) = tracks[i].state()(2) * b;
      auto cov = tracks[i].covariance();
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
    __host__ __device__ inline void copyFromDense(TrackSoAView tracks, V5 const &v, M5 const &cov, int32_t i) {
      tracks[i].state() = v.template cast<float>();
      for (int j = 0, ind = 0; j < 5; ++j)
        for (auto k = j; k < 5; ++k)
          tracks[i].covariance()(ind++) = cov(j, k);
    }

    template <typename V5, typename M5>
    __host__ __device__ inline void copyToDense(TrackSoAConstView tracks, V5 &v, M5 &cov, int32_t i) {
      v = tracks[i].state().template cast<typename V5::Scalar>();
      for (int j = 0, ind = 0; j < 5; ++j) {
        cov(j, j) = tracks[i].covariance()(ind++);
        for (auto k = j + 1; k < 5; ++k)
          cov(k, j) = cov(j, k) = tracks[i].covariance()(ind++);
      }
    }
  }  // namespace utilities
}  // namespace pixelTrack

template <int32_t S>
class TrackSoAHeterogeneousT : public cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>> {
public:
  // using cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::PortableDeviceCollection;
  TrackSoAHeterogeneousT() = default;
  explicit TrackSoAHeterogeneousT(cudaStream_t stream)
      : PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>(S, stream) {}

  explicit TrackSoAHeterogeneousT(cudaStream_t stream)
      : PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>(S, stream) {}

  static constexpr int32_t stride() { return S; }

  using Quality = pixelTrack::Quality;
  using hindex_type = uint32_t;
  using HitContainer = cms::cuda::OneToManyAssoc<hindex_type, S + 1, 5 * S>;

  // Always check quality is at least loose!
  // CUDA does not support enums  in __lgc ...
private:
public:
  // TODO: static did not work; using reinterpret_cast
  constexpr Quality const *qualityData() const { return reinterpret_cast<Quality const *>(view().quality()); }
  constexpr Quality *qualityData() { return reinterpret_cast<Quality *>(view().quality()); }

  constexpr int nHits(int i) const { return detIndices.size(i); }

  constexpr int computeNumberOfLayers(int32_t i) const {
    // layers are in order and we assume tracks are either forward or backward
    auto pdet = detIndices.begin(i);
    int nl = 1;
    auto ol = phase1PixelTopology::getLayer(*pdet);
    for (; pdet < detIndices.end(i); ++pdet) {
      auto il = phase1PixelTopology::getLayer(*pdet);
      if (il != ol)
        ++nl;
      ol = il;
    }
    return nl;
  }

  HitContainer hitIndices;
  HitContainer detIndices;
};

namespace pixelTrack {

#ifdef GPU_SMALL_EVENTS
  // kept for testing and debugging
  constexpr uint32_t maxNumber() { return 2 * 1024; }
#else
  // tested on MC events with 55-75 pileup events
  constexpr uint32_t maxNumber() { return 32 * 1024; }
#endif

  using TrackSoA = TrackSoAHeterogeneousT<maxNumber()>;
  using TrackSoAView = cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::View;
  using TrackSoAConstView = cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::ConstView;

  using HitContainer = TrackSoA::HitContainer;

}  // namespace pixelTrack

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
