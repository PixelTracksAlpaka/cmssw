#ifndef DataFormats_PixelTrackSoA_interface_PixelTrackDevice_h
#define DataFormats_PixelTrackSoA_interface_PixelTrackDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/PixelTrackSoA/interface/PixelTrackSoA.h"
#include "DataFormats/PixelTrackSoA/interface/PixelTrackDefinitions.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
// TODO: once the CUDA code is removed, this should be changed back to TrackDevice
template <typename TrackerTraits, typename TDev>
class PixelTrackDevice : public PortableDeviceCollection<PixelTrackLayout<TrackerTraits>, TDev> {
public:
  static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;  //TODO: this could be made configurable at runtime
  PixelTrackDevice() = default;                                       // necessary for ROOT dictionaries

  using PortableDeviceCollection<PixelTrackLayout<TrackerTraits>, TDev>::view;
  using PortableDeviceCollection<PixelTrackLayout<TrackerTraits>, TDev>::const_view;
  using PortableDeviceCollection<PixelTrackLayout<TrackerTraits>, TDev>::buffer;

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit PixelTrackDevice<TrackerTraits, TDev>(TQueue queue)
      : PortableDeviceCollection<PixelTrackLayout<TrackerTraits>, TDev>(S, queue) {}
};

namespace pixelTrack {

  // Not sure where these are used
  // TODO: once the CUDA code is removed, this should be changed back to TrackDevice*
  template <typename TDev>
  using PixelTrackDevicePhase1 = PixelTrackDevice<pixelTopology::Phase1, TDev>;
  template <typename TDev>
  using PixelTrackDevicePhase2 = PixelTrackDevice<pixelTopology::Phase2, TDev>;

}  // namespace pixelTrack

#endif  // DataFormats_PixelTrackSoA_interface_PixelTrackDevice_h
