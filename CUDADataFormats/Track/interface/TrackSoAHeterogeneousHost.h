#ifndef CUDADataFormats_Track_TrackHeterogeneousHost_H
#define CUDADataFormats_Track_TrackHeterogeneousHost_H

#include <cstdint>

#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/Common/interface/PortableHostCollection.h"

template <typename TrackerTraits>
class TrackSoAHeterogeneousHost : public cms::cuda::PortableHostCollection<TrackLayout<TrackerTraits>> {
public:
  TrackSoAHeterogeneousHost() = default;

  using cms::cuda::PortableHostCollection<TrackLayout<TrackerTraits>>::view;
  using cms::cuda::PortableHostCollection<TrackLayout<TrackerTraits>>::const_view;
  using cms::cuda::PortableHostCollection<TrackLayout<TrackerTraits>>::buffer;
  using cms::cuda::PortableHostCollection<TrackLayout<TrackerTraits>>::bufferSize;

  // Constructor which specifies the SoA size
  explicit TrackSoAHeterogeneousHost(cudaStream_t stream)
      : cms::cuda::PortableHostCollection<TrackLayout<TrackerTraits>>(TrackerTraits::maxNumberOfTuples, stream) {}
};

namespace pixelTrack {

  using TrackSoAHostPhase1 = TrackSoAHeterogeneousHost<pixelTopology::Phase1>;
  using TrackSoAHostPhase2 = TrackSoAHeterogeneousHost<pixelTopology::Phase2>;

}  // namespace pixelTrack

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
