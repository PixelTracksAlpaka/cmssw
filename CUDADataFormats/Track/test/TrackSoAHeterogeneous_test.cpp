/**
   Simple test of the pixelTrack::TrackSoA data structure
   which inherits from PortableDeviceCollection.

   Creates an instance of the class (automatically allocates
   memory on device), passes the view of the SoA data to
   the CUDA kernels which:
   - Fill the SoA with data.
   - Verify that the data written is correct.

 */

#include <bits/stdint-uintn.h>
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT_test.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"
#include "HeterogeneousCore/CUDAUtilities/interface/currentDevice.h"

namespace testTrackSoAHeterogeneousT {

  void runKernels(pixelTrack::TrackSoAView tracks_view);
}

int main() {
  cms::cudatest::requireDevices();

  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Inner scope to deallocate memory before destroying the stream
  {
    // Instantiate tracks on host. Portabledevicecollection allocates
    // SoA on device automatically.
    pixelTrack::TrackSoA tracks(stream);
    uint32_t soaSize = tracks.bufferSize();               // SoA Layout size (bytes)
    uint32_t soaNumElements = tracks->metadata().size();  // Length of each SoA array in elements

    // Run the tests
    testTrackSoAHeterogeneousT::runKernels(tracks.view());

    // Copy SoA data back to host
    auto tracks_h_soa = cms::cuda::make_host_unique<std::byte[]>(soaSize, stream);
    cudaCheck(cudaMemcpy(tracks_h_soa.get(), tracks.const_buffer().get(), soaSize, cudaMemcpyDeviceToHost));

    // Create a view to access the copied data
    TrackSoAHeterogeneousT_test<> tmp_layout(tracks_h_soa.get(), soaNumElements);
    TrackSoAHeterogeneousT_test<>::View tmp_view(tmp_layout);

    // Print results
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

    for (int i = 0; i < 10; ++i) {
      std::cout << tmp_view[i].pt() << "\t" << tmp_view[i].eta() << "\t" << tmp_view[i].chi2() << "\t"
                << (int)tmp_view[i].quality() << "\t" << (int)tmp_view[i].nLayers() << "\t"
                << tmp_view.hitIndices().off[i] << std::endl;
    }
  }
  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
