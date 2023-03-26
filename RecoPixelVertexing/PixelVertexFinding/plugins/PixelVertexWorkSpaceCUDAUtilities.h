#ifndef RecoPixelVertexing_PixelVertexFinding_PixelVertexWorkSpaceCUDAUtilities_h
#define RecoPixelVertexing_PixelVertexFinding_PixelVertexWorkSpaceCUDAUtilities_h

#include <cuda_runtime.h>
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "PixelVertexWorkSpaceLayout.h"

// Methods that operate on View and ConstView of the WorkSpaceSoALayout.
namespace gpuVertexFinder {
  namespace workSpace {

    namespace utilities {
      __host__ __device__ inline void init(PixelVertexWorkSpaceSoAView &workspace_view) {
        workspace_view.ntrks() = 0;
        workspace_view.nvIntermediate() = 0;
      }
    }  // namespace utilities
  }    // namespace workSpace
}  // namespace gpuVertexFinder

#endif
