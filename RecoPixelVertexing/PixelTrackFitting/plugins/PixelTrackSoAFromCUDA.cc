#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"
//#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT_test.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

// Switch on to enable checks and printout for found tracks
// #define PIXEL_DEBUG_PRODUCE

class PixelTrackSoAFromCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit PixelTrackSoAFromCUDA(const edm::ParameterSet& iConfig);
  ~PixelTrackSoAFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<pixelTrack::TrackSoAView> tokenCUDA_;
  edm::EDPutTokenT<pixelTrack::TrackSoAView> tokenSOA_;

  pixelTrack::TrackSoAView soa_view_h;
  //pixelTrack::TrackSoALayout soa_layout_h;
};

PixelTrackSoAFromCUDA::PixelTrackSoAFromCUDA(const edm::ParameterSet& iConfig)
    : tokenCUDA_(consumes<pixelTrack::TrackSoAView>(iConfig.getParameter<edm::InputTag>("src"))),
      tokenSOA_(produces<pixelTrack::TrackSoAView>()) {}

void PixelTrackSoAFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("pixelTracksCUDA"));
  descriptions.add("pixelTracksSoA", desc);
}

void PixelTrackSoAFromCUDA::acquire(edm::Event const& iEvent,
                                    edm::EventSetup const& iSetup,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  soa_view_h = iEvent.get(tokenCUDA_);
  //cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  //auto const& soa_view_h = ctx.get(inputDataWrapped);  // Layout of data on device

  /*auto soa_buffer_h = cms::cuda::make_host_unique<std::byte[]>(soa_layout_d.metadata().byteSize(), ctx.stream());

  cudaCheck(cudaMemcpyAsync(soa_buffer_h.get(),
                            soa_layout_d.metadata().data(),
                            soa_layout_d.metadata().byteSize(),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));
  pixelTrack::TrackSoALayout soa_layout_h(soa_buffer_h.get(), soa_layout_d.metadata().size());
  pixelTrack::TrackSoAView soa_view_h(soa_layout_h);*/

  // // Allocate enough host memory to fit the SoA data in the input view
  // auto soa_buffer_host = cms::cuda::make_host_unique<std::byte[]>(soa_.layout()., ctx.stream());

  // // Copy data from the view on device to host memory
  // cudaCheck(cudaMemcpy(soa_buffer_host.get(), soa_.buffer().get(), soa_.metadata().byteSize(), cudaMemcpyDeviceToHost));
  // TrackSoAHeterogeneousT_test<> soa_layout(soa_buffer_host.get(), soa_.metadata().size());
  // TrackSoAHeterogeneousT_test<>::View soa_host_view_(soa_layout);  // Store the host-side view
}

void PixelTrackSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // check that the fixed-size SoA does not overflow
  auto maxTracks = soa_view_h.metadata().size();
  auto nTracks = soa_view_h.nTracks();
  assert(nTracks < maxTracks);
  if (nTracks == maxTracks - 1) {
    edm::LogWarning("PixelTracks") << "Unsorted reconstructed pixel tracks truncated to " << maxTracks - 1
                                   << " candidates";
  }

#ifdef PIXEL_DEBUG_PRODUCE
  std::cout << " stride " << maxTracks << std::endl;
  std::cout << "found " << nTracks << std::endl;

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = pixelTrack::utilities::nHits(soa_view_h, it);
    assert(nHits == int(soa_view_h.hitIndices().size(it)));
    if (nHits == 0)
      break;  // this is a guard: maybe we need to move to nTracks...
    nt++;
  }
  assert(nTracks == nt);
#endif

  // DO NOT  make a copy  (actually TWO....)
  iEvent.emplace(tokenSOA_, std::move(soa_view_h));  //, std::move(ret)); // view

  //assert(!soa_);
}

DEFINE_FWK_MODULE(PixelTrackSoAFromCUDA);
