#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT_test.h"
#include "HeterogeneousCore/CUDAUtilities/interface/OneToManyAssoc.h"

namespace testTrackSoAHeterogeneousT {

  __global__ void fill(pixelTrack::TrackSoA* __restrict__ tracks, pixelTrack::TrackSoAView tracks_view) {
    assert(tracks);

    int i = threadIdx.x;
    for (int j = i; j < tracks->stride(); j += blockDim.x) {
      tracks_view[j].pt() = (float)j;
      tracks_view[j].eta() = (float)j;
      tracks_view[j].chi2() = (float)j;
      tracks_view[j].quality() = (uint8_t)j % 256;
      tracks_view[j].nLayers() = j % 128;
      tracks_view.hitIndices().off[j] = j;
      tracks->hitIndices.off[j] = j;
    }
  }

  __global__ void verify(pixelTrack::TrackSoA* const __restrict__ tracks, pixelTrack::TrackSoAConstView tracks_view) {
    assert(tracks);

    int i = threadIdx.x;
    if (i == 0) {
      printf("Stride: %d, block dims: %d\n", tracks->stride(), blockDim.x);
    }
    for (int j = i; j < tracks->stride(); j += blockDim.x) {
      assert(abs(tracks_view[j].pt() - (float)j) < .0001);
      assert(abs(tracks_view[j].eta() - (float)j) < .0001);
      assert(abs(tracks_view[j].chi2() - (float)j) < .0001);
      assert(tracks_view[j].quality() == j % 256);
      assert(tracks_view[j].nLayers() == j % 128);
      assert(tracks_view.hitIndices().off[j] == j);
      assert(tracks->hitIndices.off[j] == j);
    }
  }

  void runKernels(pixelTrack::TrackSoA* tracks, pixelTrack::TrackSoAView tracks_view) {
    assert(tracks);
    fill<<<1, 1024>>>(tracks, tracks_view);
    verify<<<1, 1024>>>(tracks, tracks_view);
  }

}  // namespace testTrackSoAHeterogeneousT
