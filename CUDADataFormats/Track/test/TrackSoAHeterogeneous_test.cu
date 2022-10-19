#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT_test.h"
#include "HeterogeneousCore/CUDAUtilities/interface/OneToManyAssoc.h"

namespace testTrackSoAHeterogeneousT {

  __global__ void fill(pixelTrack::TrackSoAView tracks_view) {
    int i = threadIdx.x;
    if (i == 0) {
      tracks_view.nTracks() = 420;
    }

    for (int j = i; j < tracks_view.metadata().size(); j += blockDim.x) {
      tracks_view[j].pt() = (float)j;
      tracks_view[j].eta() = (float)j;
      tracks_view[j].chi2() = (float)j;
      tracks_view[j].quality() = (uint8_t)j % 256;
      tracks_view[j].nLayers() = j % 128;
      tracks_view.hitIndices().off[j] = j;
    }
  }

  __global__ void verify(pixelTrack::TrackSoAView tracks_view) {
    int i = threadIdx.x;

    if (i == 0) {
      printf("SoA size: % d, block dims: % d\n", tracks_view.metadata().size(), blockDim.x);
      assert(tracks_view.nTracks() == 420);
    }
    for (int j = i; j < tracks_view.metadata().size(); j += blockDim.x) {
      assert(abs(tracks_view[j].pt() - (float)j) < .0001);
      assert(abs(tracks_view[j].eta() - (float)j) < .0001);
      assert(abs(tracks_view[j].chi2() - (float)j) < .0001);
      assert(tracks_view[j].quality() == j % 256);
      assert(tracks_view[j].nLayers() == j % 128);
      assert(tracks_view.hitIndices().off[j] == j);
    }
  }

  void runKernels(pixelTrack::TrackSoAView tracks_view, uint32_t soaSize) {
    fill<<<1, 1024>>>(tracks_view);
    cudaDeviceSynchronize();
    verify<<<1, 1024>>>(tracks_view);
  }

}  // namespace testTrackSoAHeterogeneousT
