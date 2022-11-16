#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoAHost.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"

#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

namespace testTrackingRecHit2DNew {

  void run(TrackingRecHitSoADevice& hits, cudaStream_t stream);

}

int main() {
  cms::cudatest::requireDevices();

  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));


  // inner scope to deallocate memory before destroying the stream
  {
    uint32_t nHits = 2000;
    int32_t offset = 100;
    uint32_t moduleStart[1856];

    for (size_t i = 0; i < 1856; i++) {
      moduleStart[i] = i*2;
    }

    TrackingRecHitSoADevice tkhit(nHits,false,offset,nullptr,&moduleStart[0],stream);

    testTrackingRecHit2DNew::run(tkhit,stream);

    auto test = tkhit.localCoordToHostAsync(stream);
    printf("tkhit hits %d \n",tkhit.nHits());
  }

  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
