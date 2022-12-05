#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"
#include "TrackingRecHitSoAImpl_t.h"

namespace testTrackingRecHitSoA {

  template<typename TrackerTraits>
  void runKernels(TrackingRecHitSoADevice<TrackerTraits>& hits, cudaStream_t stream) {
    // assert(soa);
    printf("> RUN!\n");
    fill<TrackerTraits><<<10, 100, 0, stream>>>(hits.view());

    cudaCheck(cudaDeviceSynchronize());
    cms::cuda::fillManyFromVector(hits.phiBinner(),
                                  10,
                                  hits.view().iphi(),
                                  hits.view().hitsLayerStart().data(),
                                  2000,
                                  256,
                                  hits.view().phiBinnerStorage(),
                                  stream);
    cudaCheck(cudaDeviceSynchronize());
    show<TrackerTraits><<<10, 1000, 0, stream>>>(hits.view());
    cudaCheck(cudaDeviceSynchronize());
  }

  template void runKernels<pixelTopology::Phase1>(TrackingRecHitSoADevice<pixelTopology::Phase1>& hits, cudaStream_t stream);
  template void runKernels<pixelTopology::Phase2>(TrackingRecHitSoADevice<pixelTopology::Phase2>& hits, cudaStream_t stream);

}  // namespace testTrackingRecHit2DNew
