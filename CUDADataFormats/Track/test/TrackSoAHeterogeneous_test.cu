#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT_test.h"

namespace testTrackSoAHeterogeneousT {

  __global__ void fill(pixelTrack::TrackSoAView tracks, uint32_t soaSize) {
    assert(tracks);

    int i = threadIdx.x;
    if (i > soaSize)
      return;
    tracks[i].pt() = (float) i;
  }

  __global__ void verify(pixelTrack::TrackSoAConstView tracks, uint32_t soaSize) {
    assert(tracks);

    int i = threadIdx.x;
    if (i > soaSize)
      return;
    assert(tracks[i].pt() == (float) i)
  }

  void runKernels(pixelTrack::TrackSoAView tracks, uint32_t soaSize) {
    assert(tracks);
    fill<<<1, 1024>>>(tracks, soaSize);
    verify<<<1, 1024>>>(tracks, soaSize);
  }

}  // namespace testTrackingRecHit2D
