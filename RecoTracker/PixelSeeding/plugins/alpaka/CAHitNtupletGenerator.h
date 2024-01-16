#ifndef RecoPixelVertexing_PixelTriplets_Alpaka_CAHitNtupletGenerator_h
#define RecoPixelVertexing_PixelTriplets_Alpaka_CAHitNtupletGenerator_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackerRecHitSoA/interface/TrackerRecHitSoA.h"
#include "DataFormats/TrackerRecHitSoA/interface/alpaka/TrackerRecHitSoACollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

#include "CAHitNtupletGeneratorKernels.h"
#include "CACell.h"
#include "HelixFit.h"

namespace edm {
  class ParameterSetDescription;
}  // namespace edm

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  class CAHitNtupletGenerator {
  public:
    using HitsView = TrackerRecHitSoAView<TrackerTraits>;
    using HitsConstView = TrackerRecHitSoAConstView<TrackerTraits>;
    using HitsOnDevice = TrackerRecHitSoACollection<TrackerTraits>;
    using HitsOnHost = TrackerRecHitHost<TrackerTraits>;
    using hindex_type = typename TrackerRecHitSoA<TrackerTraits>::hindex_type;

    using HitToTuple = caStructures::HitToTupleT<TrackerTraits>;
    using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;
    using OuterHitOfCell = caStructures::OuterHitOfCellT<TrackerTraits>;

    using CACell = CACellT<TrackerTraits>;
    using TkSoAHost = TracksHost<TrackerTraits>;
    using TkSoADevice = TracksSoACollection<TrackerTraits>;
    using HitContainer = typename TrackSoA<TrackerTraits>::HitContainer;
    using Tuple = HitContainer;

    using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
    using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;

    using Quality = ::pixelTrack::Quality;

    using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;
    using Params = caHitNtupletGenerator::ParamsT<TrackerTraits>;
    using Counters = caHitNtupletGenerator::Counters;

    using ParamsOnDevice = pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>;

  public:
    CAHitNtupletGenerator(const edm::ParameterSet& cfg);

    static void fillPSetDescription(edm::ParameterSetDescription& desc);
    static void fillDescriptionsCommon(edm::ParameterSetDescription& desc);

    // TODO: Check if still needed
    // void beginJob();
    // void endJob();

    TkSoADevice makeTuplesAsync(HitsOnDevice const& hits_d,
                                ParamsOnDevice const* cpeParams,
                                float bfield,
                                Queue& queue) const;

  private:
    void buildDoublets(const HitsConstView& hh, Queue& queue) const;

    void hitNtuplets(const HitsConstView& hh, const edm::EventSetup& es, bool useRiemannFit, Queue& queue);

    void launchKernels(const HitsConstView& hh, bool useRiemannFit, Queue& queue) const;

    Params m_params;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGenerator_h
