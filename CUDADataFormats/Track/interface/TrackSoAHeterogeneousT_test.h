#ifndef CUDADataFormats_Track_TrackHeterogeneousT_H
#define CUDADataFormats_Track_TrackHeterogeneousT_H

#include <string>
#include <algorithm>
#include <Eigen/Dense>
// #include "CUDADataFormats/Track/interface/TrajectoryStateSoAT.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
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

// Static helper functions for getting specific attributes from Trajectory State SoA
namespace trackSoAHeterogeneousUtilities {
  __device__ __host__ float charge(cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::ConstView tracks,
                                   int32_t i) {
    return std::copysign(1.f, tracks[i].state()(2));
  }
  __device__ __host__ float phi(cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::ConstView tracks,
                                int32_t i) {
    return tracks[i].state()(0);
  }

  __device__ __host__ float tip(cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::ConstView tracks,
                                int32_t i) {
    return tracks[i].state()(1);
  }

  __device__ __host__ float zip(cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::ConstView tracks,
                                int32_t i) {
    return tracks[i].state()(4);
  }

  __device__ __host__ bool isTriplet(
      cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::ConstView tracks, int32_t i) {
    return tracks[i].nLayers() == 3;
  }

  template <typename V3, typename M3, typename V2, typename M2>
  __host__ __device__ inline void copyFromCircle(
      cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::ConstView tracks,
      V3 const &cp,
      M3 const &ccov,
      V2 const &lp,
      M2 const &lcov,
      float b,
      int32_t i) {
    tracks[i].state() << cp.template cast<float>(), lp.template cast<float>();
    tracks[i].state()(2) *= b;
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
  __host__ __device__ inline void copyFromDense(
      cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::ConstView tracks,
      V5 const &v,
      M5 const &cov,
      int32_t i) {
    tracks[i].state() = v.template cast<float>();
    for (int j = 0, ind = 0; j < 5; ++j)
      for (auto k = j; k < 5; ++k)
        tracks[i].covariance()(ind++) = cov(j, k);
  }

  template <typename V5, typename M5>
  __host__ __device__ inline void copyToDense(
      cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::ConstView tracks,
      V5 &v,
      M5 &cov,
      int32_t i) const {
    v = tracks[i].state().template cast<typename V5::Scalar>();
    for (int j = 0, ind = 0; j < 5; ++j) {
      cov(j, j) = tracks[i].covariance()(ind++);
      for (auto k = j + 1; k < 5; ++k)
        cov(k, j) = cov(j, k) = tracks[i].covariance()(ind++);
    }
  }

}  // namespace trackSoAHeterogeneousUtilities

template <int32_t S>
class TrackSoAHeterogeneousT : public cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>> {
public:
  // using cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::PortableDeviceCollection;
  TrackSoAHeterogeneousT() = default;

  static constexpr int32_t stride() { return S; }

  using Quality = pixelTrack::Quality;
  using hindex_type = uint32_t;
  using HitContainer = cms::cuda::OneToManyAssoc<hindex_type, S + 1, 5 * S>;

  // Always check quality is at least loose!
  // CUDA does not support enums  in __lgc ...
private:
public:
  constexpr Quality quality(int32_t i) const { return static_cast<Quality>(view()[i].quality()); }
  constexpr Quality &quality(int32_t i) { return static_cast<Quality &>(view()[i].quality()); }
  // TODO: static did not work; using reinterpret_cast
  constexpr Quality const *qualityData() const { return reinterpret_cast<Quality const *>(view().quality()); }
  constexpr Quality *qualityData() { return reinterpret_cast<Quality *>(view().quality()); }

  constexpr float pt(int32_t i) const { return view()[i].pt(); }
  constexpr float &pt(int32_t i) { return view()[i].pt(); }

  constexpr float eta(int32_t i) const { return view()[i].eta(); }
  constexpr float &eta(int32_t i) { return view()[i].eta(); }

  constexpr float chi2(int32_t i) const { return view()[i].chi2(); }
  constexpr float &chi2(int32_t i) { return view()[i].chi2(); }

  constexpr int nTracks() const { return view().nTracks(); }
  constexpr void setNTracks(int n) { view().nTracks() = n; }

  constexpr int nHits(int i) const { return detIndices.size(i); }

  // constexpr bool isTriplet(int i) const { return view()[i].nLayers() == 3; }

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

  // State at the Beam spot
  // phi,tip,1/pt,cotan(theta),zip
  // constexpr float charge(int32_t i) const { return std::copysign(1.f, stateAtBS.state(i)(2)); }
  // constexpr float phi(int32_t i) const { return stateAtBS.state(i)(0); }
  // constexpr float tip(int32_t i) const { return stateAtBS.state(i)(1); }
  // constexpr float zip(int32_t i) const { return stateAtBS.state(i)(4); }

  HitContainer hitIndices;
  HitContainer detIndices;

  // private:
  //   int nTracks_;
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
  using HitContainer = TrackSoA::HitContainer;

}  // namespace pixelTrack

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
