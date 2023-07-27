#ifndef DataFormats_RecHits_interface_TrackingRecHitSoADevice_h
#define DataFormats_RecHits_interface_TrackingRecHitSoADevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoAHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsLayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TrackerTraits, typename TDev>
class TrackingRecHitAlpakaDevice : public PortableDeviceCollection<TrackingRecHitAlpakaLayout<TrackerTraits>, TDev> {
public:
  using hitSoA = TrackingRecHitAlpakaSoA<TrackerTraits>;
  //Need to decorate the class with the inherited portable accessors being now a template
  using PortableDeviceCollection<TrackingRecHitAlpakaLayout<TrackerTraits>, TDev>::view;
  using PortableDeviceCollection<TrackingRecHitAlpakaLayout<TrackerTraits>, TDev>::const_view;
  using PortableDeviceCollection<TrackingRecHitAlpakaLayout<TrackerTraits>, TDev>::buffer;

  TrackingRecHitAlpakaDevice() = default;

  using AverageGeometry = typename hitSoA::AverageGeometry;
  using ParamsOnDevice = typename hitSoA::ParamsOnDevice;
  using PhiBinnerStorageType = typename hitSoA::PhiBinnerStorageType;
  using PhiBinner = typename hitSoA::PhiBinner;
  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit TrackingRecHitAlpakaDevice(uint32_t nHits, int32_t offsetBPIX2, uint32_t const* hitsModuleStart, TQueue queue)
      : PortableDeviceCollection<TrackingRecHitAlpakaLayout<TrackerTraits>, TDev>(nHits, queue),
        nHits_(nHits),
        offsetBPIX2_(offsetBPIX2) {
    const auto host = cms::alpakatools::host();
    const auto device = cms::alpakatools::devices<alpaka::Pltf<TDev>>()[0];

    auto start_h = alpaka::createView(host, hitsModuleStart, TrackerTraits::numberOfModules + 1);
    auto start_d = alpaka::createView(device, view().hitsModuleStart().data(), TrackerTraits::numberOfModules + 1);
    alpaka::memcpy(queue, start_d, start_h);

    auto nHits_h = alpaka::createView(host, &nHits, 1);
    auto nHits_d = alpaka::createView(device, &(view().nHits()), 1);
    alpaka::memcpy(queue, nHits_d, nHits_h);

    auto off_h = alpaka::createView(host, &offsetBPIX2, 1);
    auto off_d = alpaka::createView(device, &(view().offsetBPIX2()), 1);
    alpaka::memcpy(queue, off_d, off_h);
  }

  uint32_t nHits() const { return nHits_; }  //go to size of view
  uint32_t const* hitsModuleStart() const { return view().hitsModuleStart().data(); }
  uint32_t offsetBPIX2() const { return offsetBPIX2_; }

private:
  uint32_t nHits_;  //Needed for the host SoA size
  uint32_t offsetBPIX2_;
};
#endif  // DataFormats_RecHits_interface_TrackingRecHitSoADevice_h
