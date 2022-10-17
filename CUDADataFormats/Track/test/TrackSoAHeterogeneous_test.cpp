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

  const auto soaSize = 256;
  // inner scope to deallocate memory before destroying the stream
  {
    TrackSoAHeterogeneousT<soaSize> tracks(stream);
    auto ret = cms::cuda::make_host_unique<std::byte[]>(tracks.bufferSize(), stream);
    testTrackSoAHeterogeneousT::runKernels(tracks.view(), soaSize);
    cudaCheck(cudaMemcpy(ret.get(), tracks.buffer().get(),TrackSoAHeterogeneousT_test<>::computeDataSize(soaSize),cudaMemcpyDeviceToHost));
    TrackSoAHeterogeneousT_test<> tmp_layout(ret.get(),soaSize);
    TrackSoAHeterogeneousT_test<>::View tmp_view(tmp_layout);
    std::cout << "pt" << "\t" << "eta" << "\t" <<"chi2" << "\t" << "quality" << "\t" << "nLayers" << std::endl;
    for(int i = 0; i < soaSize; ++i){
      std::cout << tmp_view[i].pt() << "\t" << tmp_view[i].eta() << "\t" << tmp_view[i].chi2() << "\t" << (int)tmp_view[i].quality() << "\t" << (int)tmp_view[i].nLayers() << std::endl;
    }
  }
  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
