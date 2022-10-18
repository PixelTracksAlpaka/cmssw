#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT_test.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"
#include "HeterogeneousCore/CUDAUtilities/interface/currentDevice.h"

namespace testTrackSoAHeterogeneousT {

  void runKernels(pixelTrack::TrackSoA *tracks, pixelTrack::TrackSoAView tracks_view);
}

int main() {
  cms::cudatest::requireDevices();

  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // inner scope to deallocate memory before destroying the stream
  {
    // Instantiate tracks on host. Portabledevicecollection allocates
    // SoA on device automatically.
    int dev = cms::cuda::currentDevice();
    pixelTrack::TrackSoA tracks_h(stream);

    // Make a copy of tracks_h to device, so that we can
    // modify hitIndices.
    void *mem = cms::cuda::allocate_device(dev, sizeof(pixelTrack::TrackSoA), stream);
    cudaCheck(cudaMemcpy(mem, &tracks_h, sizeof(pixelTrack::TrackSoA), cudaMemcpyHostToDevice));

    // Run the tests
    pixelTrack::TrackSoA *tracks_d = reinterpret_cast<pixelTrack::TrackSoA *>(mem);
    testTrackSoAHeterogeneousT::runKernels(tracks_d, tracks_h.view());

    // Copy SoA data back to host
    auto ret = cms::cuda::make_host_unique<std::byte[]>(tracks_h.bufferSize(), stream);
    cudaCheck(cudaMemcpy(ret.get(),
                         tracks_h.buffer().get(),
                         TrackSoAHeterogeneousT_test<>::computeDataSize(tracks_h.stride()),
                         cudaMemcpyDeviceToHost));

    // Copy tracks_d back to tracks_h
    cudaCheck(cudaMemcpy(&tracks_h, mem, sizeof(pixelTrack::TrackSoA), cudaMemcpyDeviceToHost));

    // Create a view to access the copied data
    TrackSoAHeterogeneousT_test<> tmp_layout(ret.get(), tracks_h.stride());
    TrackSoAHeterogeneousT_test<>::View tmp_view(tmp_layout);
    std::cout << "pt"
              << "\t"
              << "eta"
              << "\t"
              << "chi2"
              << "\t"
              << "quality"
              << "\t"
              << "nLayers"
              << "\t"
              << "hitIndices off" << std::endl;
    for (int i = 0; i < tracks_h.stride(); ++i) {
      std::cout << tmp_view[i].pt() << "\t" << tmp_view[i].eta() << "\t" << tmp_view[i].chi2() << "\t"
                << (int)tmp_view[i].quality() << "\t" << (int)tmp_view[i].nLayers() << "\t"
                << tracks_h.hitIndices.off[i] << std::endl;
    }
    cudaCheck(cudaFree(mem));
  }
  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
