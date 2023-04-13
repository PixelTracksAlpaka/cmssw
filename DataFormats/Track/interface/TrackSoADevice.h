#ifndef DataFormats_Track_TrackSoADevice_H
#define DataFormats_Track_TrackSoADevice_H

#include <cstdint>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/Track/interface/PixelTrackDefinitions.h"
#include "DataFormats/Track/interface/PixelTrackLayout.h"
#include "DataFormats/Track/interface/TrackSoAHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// TODO: The class is created via inheritance of the PortableDeviceCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
template <typename TrackerTraits, typename TDev>
class TrackSoADevice : public PortableDeviceCollection<TrackLayout<TrackerTraits>, TDev> {
public:
  static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;  //TODO: this could be made configurable at runtime
  //explicit TrackSoADevice() : PortableDeviceCollection<TrackLayout<TrackerTraits>, TDev>(S) {} //TODO: check if this is needed somewhere
  TrackSoADevice() = default;  // cms::alpakatools::Product needs this

  using PortableDeviceCollection<TrackLayout<TrackerTraits>, TDev>::view;
  using PortableDeviceCollection<TrackLayout<TrackerTraits>, TDev>::const_view;
  using PortableDeviceCollection<TrackLayout<TrackerTraits>, TDev>::buffer;

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit TrackSoADevice(TQueue queue) : PortableDeviceCollection<TrackLayout<TrackerTraits>, TDev>(S, queue) {}

  // Constructor which specifies the SoA size
  explicit TrackSoADevice(TDev const& device) : PortableDeviceCollection<TrackLayout<TrackerTraits>, TDev>(S, device) {}
};

namespace pixelTrack {
  template <typename TDev>
  using TrackSoADevicePhase1 = TrackSoADevice<pixelTopology::Phase1, TDev>;

  template <typename TDev>
  using TrackSoADevicePhase2 = TrackSoADevice<pixelTopology::Phase2, TDev>;
}  // namespace pixelTrack

#endif  // DataFormats_Track_TrackSoADevice_H