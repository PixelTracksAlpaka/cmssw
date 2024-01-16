#ifndef DataFormats_PixelTrackSoA_interface_PixelTrackHost_h
#define DataFormats_PixelTrackSoA_interface_PixelTrackHost_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/PixelTrackSoA/interface/PixelTrackSoA.h"
#include "DataFormats/PixelTrackSoA/interface/PixelTrackDefinitions.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

// TODO: The class is created via inheritance of the PortableHostCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
template <typename TrackerTraits>
class PixelTrackHost : public PortableHostCollection<PixelTrackLayout<TrackerTraits>> {
public:
  static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;  //TODO: this could be made configurable at runtime
  PixelTrackHost() = default;  // Needed for the dictionary; not sure if line above is needed anymore

  using PortableHostCollection<PixelTrackLayout<TrackerTraits>>::view;
  using PortableHostCollection<PixelTrackLayout<TrackerTraits>>::const_view;
  using PortableHostCollection<PixelTrackLayout<TrackerTraits>>::buffer;

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit PixelTrackHost<TrackerTraits>(TQueue queue) : PortableHostCollection<PixelTrackLayout<TrackerTraits>>(S, queue) {}

  // Constructor which specifies the DevHost
  explicit PixelTrackHost(alpaka_common::DevHost const& host)
      : PortableHostCollection<PixelTrackLayout<TrackerTraits>>(S, host) {}
};

namespace pixelTrack {

  using PixelTrackHostPhase1 = PixelTrackHost<pixelTopology::Phase1>;
  using PixelTrackHostPhase2 = PixelTrackHost<pixelTopology::Phase2>;
  using PixelTrackHostHIonPhase1 = PixelTrackHost<pixelTopology::HIonPhase1>;

}  // namespace pixelTrack

#endif  // DataFormats_PixelTrackSoA_interface_PixelTrackHost_h
