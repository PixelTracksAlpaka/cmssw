
#include <cstdint>
#include <memory>
#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "DataFormats/BeamSpot/interface/alpaka/BeamSpotDeviceProduct.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoADevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitSoACollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/Records/interface/PixelCPEFastParamsRecord.h"

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/alpaka/PixelCPEFastParamsCollection.h"

// #include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/Math/interface/approx_atan2.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/CommonTopologies/interface/GluedGeomDet.h"

#include "SiStripRecHitSoAKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

template <typename TrackerTraits>
class SiStripRecHitSoA : public stream::SynchronizingEDProducer<> {

  using PixelBase = typename TrackerTraits::PixelBase;

  using StripHits = TrackingRecHitAlpakaCollection<TrackerTraits>;
  using PixelHits = TrackingRecHitAlpakaCollection<PixelBase>;

  using StripHitsHost = TrackingRecHitAlpakaHost<TrackerTraits>;

  using Algo = hitkernels::SiStripRecHitSoAKernel<TrackerTraits>;

public:
  explicit SiStripRecHitSoA(const edm::ParameterSet& iConfig);
  ~SiStripRecHitSoA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:

  void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
  void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> recHitToken_;
  
  const device::EDGetToken<PixelHits> pixelRecHitSoAToken_;
  const device::EDPutToken<StripHits> stripSoA_;

  const Algo Algo_;

  StripHitsHost hits_h_;
};

template <typename TrackerTraits>
SiStripRecHitSoA<TrackerTraits>::SiStripRecHitSoA(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      recHitToken_{consumes(iConfig.getParameter<edm::InputTag>("stripRecHitSource"))},
      pixelRecHitSoAToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSoASource"))},
      stripSoA_{produces()}
{
  
}

template <typename TrackerTraits>
void SiStripRecHitSoA<TrackerTraits>::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) 
{
  std::cout << "acquire" << std::endl;
  
  // Get the objects that we need
  const TrackerGeometry* trackerGeometry = &iSetup.getData(geomToken_);

  auto const& stripHits = iEvent.get(recHitToken_);
  auto const& pixelHits = iEvent.get(pixelRecHitSoAToken_);

  // Count strip hits
  size_t nStripHits = 0;
  for (const auto& detSet : stripHits) {
    const GluedGeomDet* det = static_cast<const GluedGeomDet*>(trackerGeometry->idToDet(detSet.detId()));
    if (det->stereoDet()->index() < TrackerTraits::numberOfModules)
        nStripHits += detSet.size();
  } 

  size_t nPixelHits = pixelHits.view().metadata().size();

  std::cout << "nStripHits = " << nStripHits << std::endl;
  std::cout << "nPixelHits = " << nPixelHits << std::endl;

  // Algo_.fillWithPixels(pixelHits.view(), nStripHits);

  // Create output collection with the right size
  StripHitsHost hits_h_(
    nPixelHits + nStripHits, 
    // pixelHits.view().offsetBPIX2(),
    // pixelHits.view().hitsModuleStart().begin(),
    iEvent.queue()
  );

  alpaka::memcpy(iEvent.queue(), hits_h_.buffer(), pixelHits.buffer());

  auto& hitsModuleStart = hits_h_.view().hitsModuleStart();

  std::copy(
    pixelHits.view().hitsModuleStart().begin(),
    pixelHits.view().hitsModuleStart().end(),
    hitsModuleStart.begin()
  );

  size_t i = 0;
  size_t lastIndex = TrackerTraits::numberOfPixelModules;
  
  // Loop over strip RecHits
  for (const auto& detSet : stripHits) {

    const GluedGeomDet* det = static_cast<const GluedGeomDet*>(trackerGeometry->idToDet(detSet.detId()));
    size_t index = det->stereoDet()->index();
    
    if (index >= TrackerTraits::numberOfModules)
      break;

    // no hits since lastIndex: hitsModuleStart[lastIndex:index] = hitsModuleStart[lastIndex]
    for (auto j = lastIndex + 1; j < index + 1; ++j)
      hitsModuleStart[j] = hitsModuleStart[lastIndex];

    hitsModuleStart[index + 1] = hitsModuleStart[index] + detSet.size();
    lastIndex = index + 1;

    for (const auto& recHit : detSet) {
      hits_h_.view()[nPixelHits + i].xLocal() = recHit.localPosition().x();
      hits_h_.view()[nPixelHits + i].yLocal() = recHit.localPosition().y();
      hits_h_.view()[nPixelHits + i].xerrLocal() = recHit.localPositionError().xx();
      hits_h_.view()[nPixelHits + i].yerrLocal() = recHit.localPositionError().yy();
      auto globalPosition = det->toGlobal(recHit.localPosition());
      hits_h_.view()[nPixelHits + i].xGlobal() = globalPosition.x();
      hits_h_.view()[nPixelHits + i].yGlobal() = globalPosition.y();
      hits_h_.view()[nPixelHits + i].zGlobal() = globalPosition.z();
      hits_h_.view()[nPixelHits + i].rGlobal() = globalPosition.transverse();
      hits_h_.view()[nPixelHits + i].iphi() = unsafe_atan2s<7>(globalPosition.y(), globalPosition.x());
      // hits_h_.view()[nPixelHits + i].chargeAndStatus().charge = ?
      // hits_h_.view()[nPixelHits + i].chargeAndStatus().status = ?
      // hits_h_.view()[nPixelHits + i].clusterSizeX() = ?
      // hits_h_.view()[nPixelHits + i].clusterSizeY() = ?
      hits_h_.view()[nPixelHits + i].detectorIndex() = det->stereoDet()->index();
      // ???
      ++i;
    }
  }

  for (auto j = lastIndex + 1; j <= TrackerTraits::numberOfModules; ++j)
    hitsModuleStart[j] = hitsModuleStart[lastIndex];


  for (auto layer = 0U; layer <= TrackerTraits::numberOfLayers; ++layer) {
    hits_h_.view().hitsLayerStart()[layer] = 
      hitsModuleStart[TrackerTraits::layerStart[layer]];
  }

}
template <typename TrackerTraits>
void SiStripRecHitSoA<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("stripRecHitSource", edm::InputTag("siStripMatchedRecHits", "matchedRecHit"));
  desc.add<edm::InputTag>("pixelRecHitSoASource", edm::InputTag("siPixelRecHitsPreSplitting"));
  descriptions.addWithDefaultLabel(desc);

  // desc.setUnknown();
  // descriptions.addDefault(desc);
}

template <typename TrackerTraits>
void SiStripRecHitSoA<TrackerTraits>::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    
  iEvent.emplace(stripSoA_,Algo_.fillHitsAsync(hits_h_,iEvent.queue()));


  // Copy pixel data
  // std::copy(pixelHits.view().xLocal(), pixelHits.view().xLocal() + nPixelHits, hits_h__.view().xLocal());
  // std::copy(pixelHits.view().yLocal(), pixelHits.view().yLocal() + nPixelHits, hits_h_.view().yLocal());
  // std::copy(pixelHits.view().xerrLocal(), pixelHits.view().xerrLocal() + nPixelHits, hits_h_.view().xerrLocal());
  // std::copy(pixelHits.view().yerrLocal(), pixelHits.view().yerrLocal() + nPixelHits, hits_h_.view().yerrLocal());
  // std::copy(pixelHits.view().xGlobal(), pixelHits.view().xGlobal() + nPixelHits, hits_h_.view().xGlobal());
  // std::copy(pixelHits.view().yGlobal(), pixelHits.view().yGlobal() + nPixelHits, hits_h_.view().yGlobal());
  // std::copy(pixelHits.view().zGlobal(), pixelHits.view().zGlobal() + nPixelHits, hits_h_.view().zGlobal());
  // std::copy(pixelHits.view().rGlobal(), pixelHits.view().rGlobal() + nPixelHits, hits_h_.view().rGlobal());
  // std::copy(pixelHits.view().iphi(), pixelHits.view().iphi() + nPixelHits, hits_h_.view().iphi());
  // std::copy(pixelHits.view().chargeAndStatus(), pixelHits.view().chargeAndStatus() + nPixelHits, hits_h_.view().chargeAndStatus());
  // std::copy(pixelHits.view().clusterSizeX(), pixelHits.view().clusterSizeX() + nPixelHits, hits_h_.view().clusterSizeX());
  // std::copy(pixelHits.view().clusterSizeY(), pixelHits.view().clusterSizeY() + nPixelHits, hits_h_.view().clusterSizeY());
  // std::copy(pixelHits.view().detectorIndex(), pixelHits.view().detectorIndex() + nPixelHits, hits_h_.view().detectorIndex());

  // std::copy(pixelHits.view().phiBinnerStorage(), pixelHits.view().phiBinnerStorage() + nPixelHits, hits_h_.view().phiBinnerStorage());

  // auto& hitsModuleStart = hits_h_.view().hitsModuleStart();

  // std::copy(
  //   pixelHits.view().hitsModuleStart().begin(),
  //   pixelHits.view().hitsModuleStart().end(),
  //   hitsModuleStart.begin()
  // );

  // hits_h_.view().cpeParams() = pixelHits.view().cpeParams();
  // hits_h_.view().averageGeometry() = pixelHits.view().averageGeometry();
  // hits_h_.view().phiBinner() = pixelHits.view().phiBinner();

  // size_t i = 0;
  // size_t lastIndex = TrackerTraits::numberOfPixelModules;
  
  // // Loop over strip RecHits
  // for (const auto& detSet : stripHits) {

  //   const GluedGeomDet* det = static_cast<const GluedGeomDet*>(trackerGeometry->idToDet(detSet.detId()));
  //   size_t index = det->stereoDet()->index();
    
  //   if (index >= TrackerTraits::numberOfModules)
  //     break;

  //   // no hits since lastIndex: hitsModuleStart[lastIndex:index] = hitsModuleStart[lastIndex]
  //   for (auto j = lastIndex + 1; j < index + 1; ++j)
  //     hitsModuleStart[j] = hitsModuleStart[lastIndex];

  //   hitsModuleStart[index + 1] = hitsModuleStart[index] + detSet.size();
  //   lastIndex = index + 1;

  //   for (const auto& recHit : detSet) {
  //     hits_h_.view()[nPixelHits + i].xLocal() = recHit.localPosition().x();
  //     hits_h_.view()[nPixelHits + i].yLocal() = recHit.localPosition().y();
  //     hits_h_.view()[nPixelHits + i].xerrLocal() = recHit.localPositionError().xx();
  //     hits_h_.view()[nPixelHits + i].yerrLocal() = recHit.localPositionError().yy();
  //     auto globalPosition = det->toGlobal(recHit.localPosition());
  //     hits_h_.view()[nPixelHits + i].xGlobal() = globalPosition.x();
  //     hits_h_.view()[nPixelHits + i].yGlobal() = globalPosition.y();
  //     hits_h_.view()[nPixelHits + i].zGlobal() = globalPosition.z();
  //     hits_h_.view()[nPixelHits + i].rGlobal() = globalPosition.transverse();
  //     hits_h_.view()[nPixelHits + i].iphi() = unsafe_atan2s<7>(globalPosition.y(), globalPosition.x());
  //     // hits_h_.view()[nPixelHits + i].chargeAndStatus().charge = ?
  //     // hits_h_.view()[nPixelHits + i].chargeAndStatus().status = ?
  //     // hits_h_.view()[nPixelHits + i].clusterSizeX() = ?
  //     // hits_h_.view()[nPixelHits + i].clusterSizeY() = ?
  //     hits_h_.view()[nPixelHits + i].detectorIndex() = det->stereoDet()->index();
  //     // ???
  //     ++i;
  //   }
  // }

  // for (auto j = lastIndex + 1; j <= TrackerTraits::numberOfModules; ++j)
  //   hitsModuleStart[j] = hitsModuleStart[lastIndex];


  // for (auto layer = 0U; layer <= TrackerTraits::numberOfLayers; ++layer) {
  //   hits_h_.view().hitsLayerStart()[layer] = 
  //     hitsModuleStart[TrackerTraits::layerStart[layer]];
  // }

  // cms::alpakatools::fillManyFromVector(&(hits_h_.view().phiBinner()),
  //                               TrackerTraits::numberOfLayers,
  //                               hits_h_.view().iphi(),
  //                               hits_h_.view().hitsLayerStart().data(),
  //                               hits_h_.view().metadata().size(),
  //                               256,
  //                               hits_h_.view().phiBinnerStorage());

  
  // // auto hms = std::make_unique<uint32_t[]>(hitsModuleStart.size());
  // // std::copy(hitsModuleStart.begin(), hitsModuleStart.end(), hms.get());

  // iEvent.emplace(stripSoA_, std::move(hits_h_));
  // iEvent.emplace(moduleStartToken_, HMSstorage(std::move(hms)));
}
  using SiStripRecHitSoAPhase1 = SiStripRecHitSoA<pixelTopology::Phase1Strip>;
}

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiStripRecHitSoAPhase1);

// using SiPixelRecHitSoAFromLegacyPhase2 = SiStripRecHitSoA<pixelTopology::Phase2>;
// DEFINE_FWK_MODULE(SiPixelRecHitSoAFromLegacyPhase2);
