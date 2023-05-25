#ifndef RecoLocalTracker_SiPixelRecHits_PixelRecHitGPUKernel_h
#define RecoLocalTracker_SiPixelRecHits_PixelRecHitGPUKernel_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/BeamSpotAlpaka/interface/alpaka/BeamSpotAlpaka.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitSoADevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParams.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace pixelgpudetails {
    using namespace cms::alpakatools;

   /**
    * \class PixelRecHitGPUKernel
    *
    * \brief Sets up work division to build SoA TrackingRecHit collection from
    *        SiPixelClusters and SiPixelDigis SoAs on the device.
    *
    * The main method is makeHitsAsync() which sends a block of 128 threads for each module with digis. Each thread processes its own digi and cluster into rechit. The SoA of RecHits is created using the fillManyFromVector() method
    */

    template <typename TrackerTraits>
    class PixelRecHitGPUKernel {
    public:
      PixelRecHitGPUKernel() = default;
      ~PixelRecHitGPUKernel() = default;

      PixelRecHitGPUKernel(const PixelRecHitGPUKernel&) = delete;
      PixelRecHitGPUKernel(PixelRecHitGPUKernel&&) = delete;
      PixelRecHitGPUKernel& operator=(const PixelRecHitGPUKernel&) = delete;
      PixelRecHitGPUKernel& operator=(PixelRecHitGPUKernel&&) = delete;

      using ParamsOnGPU = pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>;

      TrackingRecHitAlpakaDevice<TrackerTraits> makeHitsAsync(SiPixelDigisDevice const& digis_d,
                                                              SiPixelClustersDevice const& clusters_d,
                                                              BeamSpotAlpaka const& bs_d,
                                                              ParamsOnGPU const* cpeParams,
                                                              Queue queue) const;
    };
  }  // namespace pixelgpudetails
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_PixelRecHitGPUKernel_h
