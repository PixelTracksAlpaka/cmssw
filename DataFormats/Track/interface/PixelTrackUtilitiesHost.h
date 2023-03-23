#ifndef DataFormats_Track_PixelTrackUtilitiesHost_h
#define DataFormats_Track_PixelTrackUtilitiesHost_h

#include "DataFormats/TrackerCommon/interface/SimplePixelTopology.h"
#include "DataFormats/Track/interface/PixelTrackDefinitions.h"
#include "DataFormats/Track/interface/PixelTrackLayout.h"

// Methods that operate on View and ConstView of the TrackSoA, and cannot be class methods.
template <typename TrackerTraits>
struct TracksUtilitiesHost {

    using TrackSoAConstView = typename TrackSoA<TrackerTraits>::template TrackSoAHeterogeneousLayout<>::ConstView;

    template <typename V5, typename M5>
    constexpr static void copyToDense(const TrackSoAConstView &tracks,
                                                                            V5 &v,
                                                                            M5 &cov,
                                                                            int32_t i) {
        v = tracks[i].state().template cast<typename V5::Scalar>();
        for (int j = 0, ind = 0; j < 5; ++j) {
        cov(j, j) = tracks[i].covariance()(ind++);
        for (auto k = j + 1; k < 5; ++k)
            cov(k, j) = cov(j, k) = tracks[i].covariance()(ind++);
        }
    }

  constexpr static int nHits(const TrackSoAConstView &tracks, int i) {
    return tracks.detIndices().size(i);
  }

  constexpr static float phi(const TrackSoAConstView &tracks, int32_t i) {
    return tracks[i].state()(0);
  }

  static float charge(const TrackSoAConstView &tracks, int32_t i) {
      return std::copysign(1.f, tracks[i].state()(2));
  }

  
};

template struct TracksUtilitiesHost<pixelTopology::Phase1>;
template struct TracksUtilitiesHost<pixelTopology::Phase2>;


#endif
