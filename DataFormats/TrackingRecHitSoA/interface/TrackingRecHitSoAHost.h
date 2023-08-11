#ifndef DataFormats_RecHits_TrackingRecHitsHost_h
#define DataFormats_RecHits_TrackingRecHitsHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsLayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TrackerTraits>
class TrackingRecHitAlpakaHost : public PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>> {
public:
  using hitSoA = TrackingRecHitAlpakaSoA<TrackerTraits>;
  //Need to decorate the class with the inherited portable accessors being now a template
  using PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>::view;
  using PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>::const_view;
  using PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>::buffer;

  TrackingRecHitAlpakaHost() = default;

  using AverageGeometry = typename hitSoA::AverageGeometry;
  using ParamsOnDevice = typename hitSoA::ParamsOnDevice;
  using PhiBinnerStorageType = typename hitSoA::PhiBinnerStorageType;
  using PhiBinner = typename hitSoA::PhiBinner;

  template <typename TQueue>
  explicit TrackingRecHitAlpakaHost(uint32_t nHits, TQueue queue)
      : PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>(nHits, queue) {}

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit TrackingRecHitAlpakaHost(uint32_t nHits, int32_t offsetBPIX2, uint32_t const* hitsModuleStart, TQueue queue)
      : PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>(nHits, queue) {
    std::copy(hitsModuleStart, hitsModuleStart + TrackerTraits::numberOfModules + 1, view().hitsModuleStart().data());
    view().offsetBPIX2() = offsetBPIX2;
  }

  uint32_t nHits() const { return view().metadata().size(); }
  uint32_t const* hitsModuleStart() const { return view().hitsModuleStart().data(); }
};

using TrackingRecHitAlpakaHostPhase1 = TrackingRecHitAlpakaHost<pixelTopology::Phase1>;
using TrackingRecHitAlpakaHostPhase2 = TrackingRecHitAlpakaHost<pixelTopology::Phase2>;
using TrackingRecHitAlpakaHostHIonPhase1 = TrackingRecHitAlpakaHost<pixelTopology::HIonPhase1>;

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
