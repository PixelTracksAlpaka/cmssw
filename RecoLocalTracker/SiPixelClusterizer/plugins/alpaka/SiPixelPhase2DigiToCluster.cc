// C++ includes
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersCollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigiErrorsCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisCollection.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelClusterThresholds.h"
#include "SiPixelRawToClusterKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelPhase2DigiToCluster : public stream::SynchronizingEDProducer<> {
  public:
    explicit SiPixelPhase2DigiToCluster(const edm::ParameterSet& iConfig);
    ~SiPixelPhase2DigiToCluster() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    using Algo = pixelDetails::SiPixelRawToClusterKernel<pixelTopology::Phase2>;

  private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    const edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> pixelDigiToken_;

    device::EDPutToken<SiPixelDigisCollection> digiPutToken_;
    device::EDPutToken<SiPixelDigiErrorsCollection> digiErrorPutToken_;
    device::EDPutToken<SiPixelClustersCollection> clusterPutToken_;

    Algo Algo_;
    
    const bool includeErrors_;
    const SiPixelClusterThresholds clusterThresholds_;
    uint32_t nDigis_ = 0;

    SiPixelDigisCollection digis_d;
  };

  SiPixelPhase2DigiToCluster::SiPixelPhase2DigiToCluster(const edm::ParameterSet& iConfig)
      : geomToken_(esConsumes()),
        pixelDigiToken_(consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("InputDigis"))),
        digiPutToken_(produces()),
        clusterPutToken_(produces()),
        includeErrors_(iConfig.getParameter<bool>("IncludeErrors")),
        clusterThresholds_{iConfig.getParameter<int32_t>("clusterThreshold_layer1"),
                           iConfig.getParameter<int32_t>("clusterThreshold_otherLayers")} {
    if (includeErrors_) {
      digiErrorPutToken_ = produces();
    }
  }

  void SiPixelPhase2DigiToCluster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<bool>("IncludeErrors", true);
    desc.add<int32_t>("clusterThreshold_layer1", pixelClustering::clusterThresholdLayerOne); //FIXME (fix the CUDA)
    desc.add<int32_t>("clusterThreshold_otherLayers", pixelClustering::clusterThresholdOtherLayers);
    desc.add<edm::InputTag>("InputDigis", edm::InputTag("simSiPixelDigis:Pixel"));
    descriptions.addWithDefaultLabel(desc);
  }

  void SiPixelPhase2DigiToCluster::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    auto const& input = iEvent.get(pixelDigiToken_);

    const TrackerGeometry* geom_ = &iSetup.getData(geomToken_);

    uint32_t nDigis = 0;
    for (auto DSViter = input.begin(); DSViter != input.end(); DSViter++) {
      nDigis = nDigis + DSViter->size();
    }
    SiPixelDigisHost digis_h(nDigis, iEvent.queue());
    nDigis_ = nDigis;

    if (nDigis_ == 0)
      return;

    nDigis = 0;
    for (auto DSViter = input.begin(); DSViter != input.end(); DSViter++) {
      unsigned int detid = DSViter->detId();
      DetId detIdObject(detid);
      const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
      auto const gind = genericDet->index();
      for (auto const& px : *DSViter) {
        digis_h.view()[nDigis].moduleId() = uint16_t(gind);

        digis_h.view()[nDigis].xx() = uint16_t(px.row());
        digis_h.view()[nDigis].yy() = uint16_t(px.column());
        digis_h.view()[nDigis].adc() = uint16_t(px.adc());

        digis_h.view()[nDigis].pdigi() = uint32_t(px.packedData());

        digis_h.view()[nDigis].rawIdArr() = uint32_t(detid);

        nDigis++;
      }
    }

    digis_d = SiPixelDigisCollection(nDigis, iEvent.queue());
    alpaka::memcpy(iEvent.queue(), digis_d.buffer(), digis_h.buffer());

    Algo_.makePhase2ClustersAsync(clusterThresholds_, digis_d.view(), nDigis, iEvent.queue());
  }

  void SiPixelPhase2DigiToCluster::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    if (nDigis_ == 0) {
      SiPixelClustersCollection clusters_d{pixelTopology::Phase1::numberOfModules, iEvent.queue()};
      iEvent.emplace(digiPutToken_, std::move(digis_d));
      iEvent.emplace(clusterPutToken_, std::move(clusters_d));
      if (includeErrors_) {
        iEvent.emplace(digiErrorPutToken_, SiPixelDigiErrorsCollection());
      }
      return;
    }

    iEvent.emplace(digiPutToken_, std::move(digis_d));
    iEvent.emplace(clusterPutToken_, Algo_.getClusters());
    if (includeErrors_) {
      iEvent.emplace(digiErrorPutToken_, Algo_.getErrors());
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// define as framework plugin
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiPixelPhase2DigiToCluster);
