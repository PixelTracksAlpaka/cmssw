#ifndef CUDADataFormats_Track_TrackHeterogeneousHost_H
#define CUDADataFormats_Track_TrackHeterogeneousHost_H

#include <bits/stdint-uintn.h>

#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
//#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
//#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"
//#include "HeterogeneousCore/CUDAUtilities/interface/allocate_host.h"

template <int32_t S>
class TrackSoAHeterogeneousHost : public cms::cuda::PortableHostCollection<TrackSoAHeterogeneousLayout<>> {
public:
  TrackSoAHeterogeneousHost() = default;

  // Constructor which specifies the SoA size
  explicit TrackSoAHeterogeneousHost(cudaStream_t stream)
      : PortableHostCollection<TrackSoAHeterogeneousLayout<>>(S, stream) {}



};

namespace pixelTrack {

  using TrackSoAHost = TrackSoAHeterogeneousHost<maxNumber()>;

}  // namespace pixelTrack

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
