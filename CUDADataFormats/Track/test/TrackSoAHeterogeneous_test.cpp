#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT_test.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

namespace testTrackSoAHeterogeneousT {

  void runKernels(pixelTrack::TrackSoAView tracks, uint32_t soaSize);

}

int main() {
  cms::cudatest::requireDevices();

  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  auto soaSize = 200;
  // inner scope to deallocate memory before destroying the stream
  {
    /*TrackingRecHit2DGPU tkhit(nHits, false, 0, nullptr, nullptr, stream);
    testTrackingRecHit2D::runKernels(tkhit.view());

    TrackingRecHit2DGPU tkhitPhase2(nHits, true, 0, nullptr, nullptr, stream);
    testTrackingRecHit2D::runKernels(tkhitPhase2.view());

    TrackingRecHit2DHost tkhitH(nHits, false, 0, nullptr, nullptr, stream, &tkhit);
    cudaStreamSynchronize(stream);
    assert(tkhitH.view());
    assert(tkhitH.view()->nHits() == unsigned(nHits));
    assert(tkhitH.view()->nMaxModules() == phase1PixelTopology::numberOfModules);

    TrackingRecHit2DHost tkhitHPhase2(nHits, true, 0, nullptr, nullptr, stream, &tkhit);
    cudaStreamSynchronize(stream);
    assert(tkhitHPhase2.view());
    assert(tkhitHPhase2.view()->nHits() == unsigned(nHits));
    assert(tkhitHPhase2.view()->nMaxModules() == phase2PixelTopology::numberOfModules);*/

    pixelTrack::TrackSoA tracks;
    testTrackSoAHeterogeneousT::runKernels(tracks.view(), soaSize);
    std::cout << typeid(tracks.view()).name() << std::endl;
  }

  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
