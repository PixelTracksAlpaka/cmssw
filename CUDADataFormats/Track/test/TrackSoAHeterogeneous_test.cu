#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT_test.h"

namespace testTrackSoAHeterogeneousT {

  __global__ void fill(pixelTrack::TrackSoAView tracks, uint32_t soaSize) {
    //assert(tracks);

    int i = threadIdx.x;
    if (i >= soaSize)
      return;
    tracks[i].pt() = (float)i;
    tracks[i].eta() = (float)i;
    tracks[i].chi2() = (float)i;
    tracks[i].quality() = (uint8_t)i;
    tracks[i].nLayers() = i % 128;
  }

  __global__ void verify(pixelTrack::TrackSoAConstView tracks, uint32_t soaSize) {
    //assert(tracks);

    int i = threadIdx.x;
    if (i >= soaSize)
      return;
    assert(abs(tracks[i].pt() - (float)i) < .0001);
    assert(abs(tracks[i].eta() - (float)i) < .0001);
    assert(abs(tracks[i].chi2() - (float)i) < .0001);
    assert(tracks[i].quality() == i);
    assert(tracks[i].nLayers() == i % 128);
  }

  void runKernels(pixelTrack::TrackSoAView tracks, uint32_t soaSize) {
    //assert(tracks);
    fill<<<1, 1024>>>(tracks, soaSize);
    verify<<<1, 1024>>>(tracks, soaSize);
  }

}  // namespace testTrackingRecHit2D
