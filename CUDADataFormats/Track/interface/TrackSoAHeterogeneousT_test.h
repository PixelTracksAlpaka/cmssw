#ifndef CUDADataFormats_Track_TrackHeterogeneousT_H
#define CUDADataFormats_Track_TrackHeterogeneousT_H

#include <string>
#include <algorithm>

#include "CUDADataFormats/Track/interface/TrajectoryStateSoAT.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

//#include "DataFormats/Portable/interface/PortableCUDADeviceCollection.h"
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

GENERATE_SOA_LAYOUT(TrackSoAHeterogeneousT_test,
                    SOA_COLUMN(uint8_t, quality),
                    SOA_COLUMN(float, chi2),  // this is chi2/ndof as not necessarely all hits are used in the fit
                    SOA_COLUMN(int8_t, nLayers),
                    SOA_COLUMN(float, eta),
                    SOA_COLUMN(float, pt))
// TODO: maybe add stateAtBS

template <int32_t S>
class TrackSoAHeterogeneousT : public cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>> {
public:
  // using cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>::PortableDeviceCollection;
  TrackSoAHeterogeneousT() = default;
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
  constexpr Quality quality(int32_t i) const { return static_cast<Quality>(view()[i].quality()); }
  // constexpr Quality &quality(int32_t i) { return static_cast<Quality &>(view()[i].quality()); }
  // TODO: static did not work; using reinterpret_cast
  constexpr Quality const *qualityData() const { return reinterpret_cast <Quality const *>(view().quality()); }
  constexpr Quality *qualityData() { return reinterpret_cast< Quality *>(view().quality()); }

  constexpr float pt(int32_t i) const { return view()[i].pt(); }
  // constexpr float &pt(int32_t i) { return view()[i].pt(); }

  constexpr float eta(int32_t i) const { return view()[i].eta(); }
  // constexpr float &eta(int32_t i) { return view()[i].eta(); }

  constexpr float chi2(int32_t i) const { return view()[i].chi2(); }
  // constexpr float &chi2(int32_t i) { return view()[i].chi2(); }

  constexpr int nTracks() const { return nTracks_; }
  constexpr void setNTracks(int n) { nTracks_ = n; }

  constexpr int nHits(int i) const { return detIndices.size(i); }

  constexpr bool isTriplet(int i) const { return view()[i].nLayers() == 3; }

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
  TrajectoryStateSoAT<S> stateAtBS;
  constexpr float charge(int32_t i) const { return std::copysign(1.f, stateAtBS.state(i)(2)); }
  constexpr float phi(int32_t i) const { return stateAtBS.state(i)(0); }
  constexpr float tip(int32_t i) const { return stateAtBS.state(i)(1); }
  constexpr float zip(int32_t i) const { return stateAtBS.state(i)(4); }

  // state at the detector of the outermost hit
  // representation to be decided...
  // not yet filled on GPU
  // TrajectoryStateSoA<S> stateAtOuterDet;

  HitContainer hitIndices;
  HitContainer detIndices;

private:
  int nTracks_;
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

  using TrajectoryState = TrajectoryStateSoAT<maxNumber()>;
  using HitContainer = TrackSoA::HitContainer;

}  // namespace pixelTrack

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
