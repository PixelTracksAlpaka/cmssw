#ifndef CUDADataFormats_Track_TrackHeterogeneousDevice_H
#define CUDADataFormats_Track_TrackHeterogeneousDevice_H

#include <bits/stdint-uintn.h>

#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

template <int32_t S>
class TrackSoAHeterogeneousDevice : public cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousLayout<>> {
public:
  TrackSoAHeterogeneousDevice() = default;  // cms::cuda::Product needs this

  // Constructor which specifies the SoA size
  explicit TrackSoAHeterogeneousDevice(cudaStream_t stream)
      : PortableDeviceCollection<TrackSoAHeterogeneousLayout<>>(S, stream) {}

  // Copy data from device to host
  __host__ void copyToHost(cms::cuda::host::unique_ptr<std::byte[]> &host_ptr, cudaStream_t stream) {
    cudaCheck(cudaMemcpy(host_ptr.get(), const_buffer().get(), bufferSize(), cudaMemcpyDeviceToHost));
  }
};

namespace pixelTrack {

  using TrackSoADevice = TrackSoAHeterogeneousDevice<pixelTrack::maxNumber()>;

}  // namespace pixelTrack

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
