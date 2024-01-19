#ifndef DataFormats_PixelTrackSoA_interface_PixelTrackSoA_h
#define DataFormats_PixelTrackSoA_interface_PixelTrackSoA_h

#include <Eigen/Core>
#include "HeterogeneousCore/AlpakaInterface/interface/OneToManyAssoc.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/PixelTrackSoA/interface/PixelTrackDefinitions.h"

template <typename TrackerTraits>
struct PixelTrackSoA { // TODO: once the CUDA code is removed, this should be changed back to TrackSoA
  static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;
  static constexpr int32_t H = TrackerTraits::avgHitsPerTrack;
  // Aliases in order to not confuse the GENERATE_SOA_LAYOUT
  // macro with weird colons and angled brackets.
  using Vector5f = Eigen::Matrix<float, 5, 1>;
  using Vector15f = Eigen::Matrix<float, 15, 1>;
  using Quality = pixelTrackSoA::Quality;

  using hindex_type = uint32_t;

  using HitContainer = cms::alpakatools::OneToManyAssocSequential<hindex_type, S + 1, H * S>;

  GENERATE_SOA_LAYOUT(Layout,
                      SOA_COLUMN(Quality, quality),
                      SOA_COLUMN(float, chi2),
                      SOA_COLUMN(int8_t, nLayers),
                      SOA_COLUMN(float, eta),
                      SOA_COLUMN(float, pt),
                      SOA_EIGEN_COLUMN(Vector5f, state),
                      SOA_EIGEN_COLUMN(Vector15f, covariance),
                      SOA_SCALAR(int, nTracks),
                      SOA_SCALAR(HitContainer, hitIndices),
                      SOA_SCALAR(HitContainer, detIndices))
};

// TODO: once the CUDA code is removed, this should be changed back to Track*
template <typename TrackerTraits>
using PixelTrackLayout = typename PixelTrackSoA<TrackerTraits>::template Layout<>;
template <typename TrackerTraits>
using PixelTrackSoAView = typename PixelTrackSoA<TrackerTraits>::template Layout<>::View;
template <typename TrackerTraits>
using PixelTrackSoAConstView = typename PixelTrackSoA<TrackerTraits>::template Layout<>::ConstView;

#endif  // DataFormats_PixelTrackSoA_interface_PixelTrackSoA_h
