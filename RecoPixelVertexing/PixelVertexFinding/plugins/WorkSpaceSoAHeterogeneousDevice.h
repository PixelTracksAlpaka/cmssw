#ifndef RecoPixelVertexing_PixelVertexFinding_WorkSpaceSoAHeterogeneousDevice_h
#define RecoPixelVertexing_PixelVertexFinding_WorkSpaceSoAHeterogeneousDevice_h

#include <cstdint>
#include "WorkSpaceUtilities.h"
#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "CUDADataFormats/Vertex/interface/WorkSpaceUtilities.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"

template <int32_t S>
class WorkSpaceSoAHeterogeneousDevice : public cms::cuda::PortableDeviceCollection<WorksSpaceSoAHeterogeneousLayout> {
  WorkSpaceSoAHeterogeneousDevice() = default;

  // Constructor which specifies the SoA size and CUDA stream
  explicit WorkSpaceSoAHeterogeneousDevice(cudaStream_t stream)
      : PortableDeviceCollection<WorkSpaceSoAHeterogeneousLayout<>>(S, stream) {}
};

namespace gpuVertexFinder {
  namespace WorkSpace {
    using WorkSpaceSoADevice = WorkSpaceSoAHeterogeneousDevice<ZVertex::utilities::MAXTRACKS>;
  }
}  // namespace gpuVertexFinder
#endif
