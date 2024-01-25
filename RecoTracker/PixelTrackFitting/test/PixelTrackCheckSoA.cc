#include <Eigen/Core>  // needed here by soa layout

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"

template <typename TrackerTraits>
class PixelTrackCheckSoAT : public edm::global::EDAnalyzer<> {
public:
  using TkSoAHost = TracksHost<TrackerTraits>;
  using VertexSoAHost = ZVertexHost;

  explicit PixelTrackCheckSoAT(const edm::ParameterSet& iConfig);
  ~PixelTrackCheckSoAT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID streamID, edm::Event const& iEvent, const edm::EventSetup& iSetup) const override;
  edm::EDGetTokenT<TkSoAHost> tokenSoATrack_;
  edm::EDGetTokenT<VertexSoAHost> tokenSoAVertex_;
};

template <typename TrackerTraits>
PixelTrackCheckSoAT<TrackerTraits>::PixelTrackCheckSoAT(const edm::ParameterSet& iConfig) {
  tokenSoATrack_ = consumes(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"));
  tokenSoAVertex_ = consumes(iConfig.getParameter<edm::InputTag>("pixelVertexSrc"));
}

template <typename TrackerTraits>
void PixelTrackCheckSoAT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("pixelTracksAlpaka"));
  desc.add<edm::InputTag>("pixelVertexSrc", edm::InputTag("pixelVerticesAlpaka"));
  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void PixelTrackCheckSoAT<TrackerTraits>::analyze(edm::StreamID streamID,
                                                   edm::Event const& iEvent,
                                                   const edm::EventSetup& iSetup) const {
  auto const& tracks = iEvent.get(tokenSoATrack_);
  assert(tracks.view().quality());
  assert(tracks.view().chi2());
  assert(tracks.view().nLayers());
  assert(tracks.view().eta());
  assert(tracks.view().pt());
  assert(tracks.view().state());
  assert(tracks.view().covariance());
  assert(tracks.view().nTracks());

  auto const& vertices = iEvent.get(tokenSoAVertex_);
  assert(vertices.view().idv());
  assert(vertices.view().zv());
  assert(vertices.view().wv());
  assert(vertices.view().chi2());
  assert(vertices.view().ptv2());
  assert(vertices.view().ndof());
  assert(vertices.view().sortInd());
  assert(vertices.view().nvFinal());
}
using PixelTrackCheckSoAPhase1 = PixelTrackCheckSoAT<pixelTopology::Phase1>;
using PixelTrackCheckSoAPhase2 = PixelTrackCheckSoAT<pixelTopology::Phase2>;

DEFINE_FWK_MODULE(PixelTrackCheckSoAPhase1);
DEFINE_FWK_MODULE(PixelTrackCheckSoAPhase2);
