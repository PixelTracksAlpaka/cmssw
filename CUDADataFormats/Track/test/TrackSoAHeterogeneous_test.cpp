#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT_test.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

namespace testTrackSoAHeterogeneousT {
  void runKernels(pixelTrack::TrackSoAView tracks, unsigned int soaSize);
}

int main() {
  cms::cudatest::requireDevices();

  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const auto soaSize = 256;
  // inner scope to deallocate memory before destroying the stream
  {
    pixelTrack::TrackSoA tracks;
    testTrackSoAHeterogeneousT::runKernels(tracks.view(), soaSize);
  }
  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
