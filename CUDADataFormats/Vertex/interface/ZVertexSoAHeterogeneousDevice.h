#ifndef CUDADataFormats_Vertex_ZVertexHeterogeneousDevice_H
#define CUDADataFormats_Vertex_ZVertexHeterogeneousDevice_H

#include <bits/stdint-uintn.h>

#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

template <int32_t S>
class ZVertexSoAHeterogeneousDevice : public cms::cuda::PortableDeviceCollection<ZVertexSoAHeterogeneousLayout<>> {
public:
  ZVertexSoAHeterogeneousDevice() = default;  // cms::cuda::Product needs this

  // Constructor which specifies the SoA size
  explicit ZVertexSoAHeterogeneousDevice(cudaStream_t stream)
      : PortableDeviceCollection<ZVertexSoAHeterogeneousLayout<>>(S, stream) {}

  // Copy data from device to host
  __host__ void copyToHost(cms::cuda::host::unique_ptr<std::byte[]> &host_ptr, cudaStream_t stream) const {
    cudaCheck(cudaMemcpyAsync(host_ptr.get(), const_buffer().get(), bufferSize(), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaGetLastError());
  }
};

namespace ZVertex {

  using ZVertexSoADevice = ZVertexSoAHeterogeneousDevice<ZVertex::utilities::MAXTRACKS>;

}  // namespace pixelTrack

#endif  // CUDADataFormats_Vertex_ZVertexHeterogeneousT_H
