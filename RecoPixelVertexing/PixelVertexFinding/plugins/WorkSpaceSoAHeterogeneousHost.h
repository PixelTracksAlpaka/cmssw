#ifndef RecoPixelVertexing_PixelVertexFinding_WorkSpaceSoAHeterogeneousHost_h
#define RecoPixelVertexing_PixelVertexFinding_WorkSpaceSoAHeterogeneousHost_h

#include <cstdint>
#include "WorkSpaceUtilities.h"
#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "CUDADataFormats/Vertex/interface/WorkSpaceUtilities.h"
#include "CUDADataFormats/Common/interface/PortableHostCollection.h"

template <int32_t S>
class WorkSpaceSoAHeterogeneousHost : public cms::cuda::PortableHostCollection<WorksSpaceSoAHeterogeneousLayout> {
  WorkSpaceSoAHeterogeneousHost() = default;

  // Constructor which specifies the SoA size and CUDA stream
  explicit WorkSpaceSoAHeterogeneousHost(cudaStream_t stream)
      : PortableHostCollection<WorkSpaceSoAHeterogeneousLayout<>>(S, stream) {}
};

namespace gpuVertexFinder {
  namespace WorkSpace {
    using WorkSpaceSoAHost = WorkSpaceSoAHeterogeneousHost<ZVertex::utilities::MAXTRACKS>;
  }
}  // namespace gpuVertexFinder
#endif
