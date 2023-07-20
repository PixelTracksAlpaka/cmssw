#ifndef RecoLocalTracker_SiPixelRecHits_PixelRecHitGPUKernel_h
#define RecoLocalTracker_SiPixelRecHits_PixelRecHitGPUKernel_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersCollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoADevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitSoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

/*! \file PixelRecHitGPUKernel.h
 * ------------------------------------------------------
 * Defines a class that sends jobs to device
 * creating SoA of RecHits from SiPixelClusters and SiPixelDigis collections
 * ------------------------------------------------------
 */

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace pixelgpudetails {
    using namespace cms::alpakatools;


    //! \class PixelRecHitGPUKernel
    //!
    //! \brief Sets up work division to build SoA TrackingRecHit collection from SiPixelClusters and SiPixelDigis SoAs on the device.
    template <typename TrackerTraits>
    class PixelRecHitGPUKernel {
    public:
      PixelRecHitGPUKernel() = default;
      ~PixelRecHitGPUKernel() = default;

      PixelRecHitGPUKernel(const PixelRecHitGPUKernel&) = delete;
      PixelRecHitGPUKernel(PixelRecHitGPUKernel&&) = delete;
      PixelRecHitGPUKernel& operator=(const PixelRecHitGPUKernel&) = delete;
      PixelRecHitGPUKernel& operator=(PixelRecHitGPUKernel&&) = delete;

      using ParamsOnDevice = pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>;

      //! \fn makeHitsAsync
      //!
      //! Sends a block of 128 threads for each module with digis. Each thread processes its own digi and cluster into rechit.
      //! The SoA of RecHits is created using fillManyFromVector method.
      TrackingRecHitAlpakaCollection<TrackerTraits> makeHitsAsync(SiPixelDigisCollection const& digis_d,
                                                              SiPixelClustersCollection const& clusters_d,
                                                              BeamSpotPOD const* bs_d,
                                                              ParamsOnDevice const* cpeParams,
                                                              Queue queue) const;
    };
  }  // namespace pixelgpudetails
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_PixelRecHitGPUKernel_h
