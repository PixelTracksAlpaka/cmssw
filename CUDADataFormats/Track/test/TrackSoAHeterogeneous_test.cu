#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "HeterogeneousCore/CUDAUtilities/interface/OneToManyAssoc.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "TrackSoAHeterogeneousImpl_test.h"

namespace testTrackSoA {

  // Host function which invokes the two kernels above
  template< typename TrackerTraits>
  void runKernels(TrackSoAView<TrackerTraits> tracks_view, cudaStream_t stream) {
    fill<TrackerTraits><<<1, 1024, 0, stream>>>(tracks_view);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    verify<TrackerTraits><<<1, 1024, 0, stream>>>(tracks_view);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
  }

  template void runKernels<pixelTopology::Phase1>(TrackSoAView<pixelTopology::Phase1> tracks_view, cudaStream_t stream);
  template void runKernels<pixelTopology::Phase2>(TrackSoAView<pixelTopology::Phase2> tracks_view, cudaStream_t stream);

}  // namespace testTrackSoAHeterogeneousT
