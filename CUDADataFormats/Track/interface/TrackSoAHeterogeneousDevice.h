#ifndef CUDADataFormats_Track_TrackHeterogeneousDevice_H
#define CUDADataFormats_Track_TrackHeterogeneousDevice_H

#include <bits/stdint-uintn.h>

#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
//#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
//#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"
//#include "HeterogeneousCore/CUDAUtilities/interface/allocate_host.h"

template <int32_t S>
class TrackSoAHeterogeneousDevice : public cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousLayout<>> {
public:
  //TrackSoAHeterogeneousDevice() = default;

  // Constructor which specifies the SoA size
  explicit TrackSoAHeterogeneousDevice(cudaStream_t stream)
      : PortableDeviceCollection<TrackSoAHeterogeneousLayout<>>(S, stream) {}

  // Copy data from device to host
  __host__ void copyToHost(cms::cuda::host::unique_ptr<std::byte[]> &host_ptr, cudaStream_t stream) {
    cudaCheck(cudaMemcpy(host_ptr.get(), const_buffer().get(), bufferSize(), cudaMemcpyDeviceToHost));
  }
};

namespace pixelTrack {

  using TrackSoADevice = TrackSoAHeterogeneousDevice<maxNumber()>;

}  // namespace pixelTrack

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
