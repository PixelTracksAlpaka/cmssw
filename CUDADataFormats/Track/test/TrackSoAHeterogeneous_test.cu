#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT_test.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
namespace testTrackSoAHeterogeneousT {

  __global__ void fill(pixelTrack::TrackSoAView tracks, unsigned int soaSize) {
    int i = threadIdx.x;
    if (i >= soaSize)
      return;
    tracks[i].pt() = (float)i;
  }

  __global__ void verify(pixelTrack::TrackSoAConstView tracks, unsigned int soaSize) {
    int i = threadIdx.x;
    if (i >= soaSize)
      return;
    assert(tracks[i].pt() == (float)i);
  }

  void runKernels(pixelTrack::TrackSoAView tracks, unsigned int soaSize) {
    fill<<<1, 1024>>>(tracks, soaSize);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
      printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    verify<<<1, 1024>>>(tracks, soaSize);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
      printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
  }

}  // namespace testTrackSoAHeterogeneousT
