#ifndef DataFormats_TrackerRecHitSoA_interface_TrackerRecHitHost_h
#define DataFormats_TrackerRecHitSoA_interface_TrackerRecHitHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TrackerRecHitSoA/interface/TrackerRecHitSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TrackerTraits>
class TrackerRecHitHost : public PortableHostCollection<TrackerRecHitLayout<TrackerTraits>> {
public:
  using hitSoA = TrackerRecHitSoA<TrackerTraits>;
  //Need to decorate the class with the inherited portable accessors being now a template
  using PortableHostCollection<TrackerRecHitLayout<TrackerTraits>>::view;
  using PortableHostCollection<TrackerRecHitLayout<TrackerTraits>>::const_view;
  using PortableHostCollection<TrackerRecHitLayout<TrackerTraits>>::buffer;

  TrackerRecHitHost() = default;

  template <typename TQueue>
  explicit TrackerRecHitHost(uint32_t nHits, TQueue queue)
      : PortableHostCollection<TrackerRecHitLayout<TrackerTraits>>(nHits, queue) {}

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit TrackerRecHitHost(uint32_t nHits, int32_t offsetBPIX2, uint32_t const* hitsModuleStart, TQueue queue)
      : PortableHostCollection<TrackerRecHitLayout<TrackerTraits>>(nHits, queue) {
    std::copy(hitsModuleStart, hitsModuleStart + TrackerTraits::numberOfModules + 1, view().hitsModuleStart().data());
    view().offsetBPIX2() = offsetBPIX2;
  }

  uint32_t nHits() const { return view().metadata().size(); }
  uint32_t const* hitsModuleStart() const { return view().hitsModuleStart().data(); }
};

using TrackerRecHitHostPhase1 = TrackerRecHitHost<pixelTopology::Phase1>;
using TrackerRecHitHostPhase2 = TrackerRecHitHost<pixelTopology::Phase2>;
using TrackerRecHitHostHIonPhase1 = TrackerRecHitHost<pixelTopology::HIonPhase1>;

#endif  // DataFormats_TrackerRecHitSoA_interface_TrackerRecHitHost_h
