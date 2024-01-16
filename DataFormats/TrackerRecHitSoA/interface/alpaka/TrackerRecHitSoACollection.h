#ifndef DataFormats_TrackerRecHitSoA_interface_alpaka_TrackerRecHitSoACollection_h
#define DataFormats_TrackerRecHitSoA_interface_alpaka_TrackerRecHitSoACollection_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackerRecHitSoA/interface/TrackerRecHitSoA.h"
#include "DataFormats/TrackerRecHitSoA/interface/TrackerRecHitHost.h"
#include "DataFormats/TrackerRecHitSoA/interface/TrackerRecHitDevice.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  using TrackerRecHitSoACollection = std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>,
                                                          TrackerRecHitHost<TrackerTraits>,
                                                          TrackerRecHitDevice<TrackerTraits, Device>>;

  //Classes definition for Phase1/Phase2, to make the classes_def lighter. Not actually used in the code.
  using TrackerRecHitSoAPhase1 = TrackerRecHitSoACollection<pixelTopology::Phase1>;
  using TrackerRecHitSoAPhase2 = TrackerRecHitSoACollection<pixelTopology::Phase2>;
  using TrackerRecHitSoAHIonPhase1 = TrackerRecHitSoACollection<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TrackerTraits, typename TDevice>
  struct CopyToHost<TrackerRecHitDevice<TrackerTraits, TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, TrackerRecHitDevice<TrackerTraits, TDevice> const& deviceData) {
      TrackerRecHitHost<TrackerTraits> hostData(deviceData.view().metadata().size(), queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
#ifdef GPU_DEBUG
      printf("TrackerRecHitSoACollection: I'm copying to host.\n");
#endif
      return hostData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(TrackerRecHitSoAPhase1, TrackerRecHitHostPhase1);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(TrackerRecHitSoAPhase2, TrackerRecHitHostPhase2);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(TrackerRecHitSoAHIonPhase1, TrackerRecHitHostHIonPhase1);

#endif  // DataFormats_TrackerRecHitSoA_interface_alpaka_TrackerRecHitSoACollection_h