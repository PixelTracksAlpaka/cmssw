#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"

namespace testTrackingRecHitSoA {


  template<typename TrackerTraits>
  __global__ void fill(HitSoAView<TrackerTraits> soa) {

    int i = threadIdx.x;
    int j = blockIdx.x;
    if (i == 0 and j == 0) {
      soa.offsetBPIX2() = 22;
      soa[10].xLocal() = 1.11;
    }

    soa[i].iphi() = i % 10;
    soa.hitsLayerStart()[j] = j;
    __syncthreads();
  }

  template <typename TrackerTraits>
  __global__ void show(HitSoAView<TrackerTraits> soa) {

    int i = threadIdx.x;
    int j = blockIdx.x;

    if (i == 0 and j == 0) {
      printf("nbins = %d \n", soa.phiBinner().nbins());
      printf("offsetBPIX %d ->%d \n", i, soa.offsetBPIX2());
      printf("nHits %d ->%d \n", i, soa.nHits());
      printf("hitsModuleStart %d ->%d \n", i, soa.hitsModuleStart().at(28));
    }

    if (i < soa.nHits())
      printf("iPhi %d ->%d \n", i, soa[i].iphi());

    if (j * blockDim.x + i < soa.phiBinner().nbins())
      printf(">bin size %d ->%d \n", j * blockDim.x + i, soa.phiBinner().size(j * blockDim.x + i));
    __syncthreads();
  }

}  // namespace testTrackingRecHit2D
