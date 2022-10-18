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

  //edm::EDGetTokenT<cms::cuda::Product<PixelTrackHeterogeneous>> tokenCUDA_;
  //edm::EDPutTokenT<PixelTrackHeterogeneous> tokenSOA_;

  //edm::EDGetTokenT<cms::cuda::Product<TrackSoAHeterogeneousT<32768>>> tokenCUDA_;
  edm::EDGetTokenT<cms::cuda::Product<pixelTrack::TrackSoA>> tokenCUDA_;
  //edm::EDPutTokenT<TrackSoAHeterogeneousT_test<>::View> tokenSOA_;
  edm::EDPutTokenT<pixelTrack::TrackSoA> tokenSOA_;

  //cms::cuda::host::unique_ptr<pixelTrack::TrackSoA> soa_;
  //cms::cuda::host::unique_ptr<pixelTrack::TrackSoA> soa_;
  //TrackSoAHeterogeneousT_test<>::View soa_;
  pixelTrack::TrackSoA soa_;
  pixelTrack::TrackSoAView tmp_view_;
};

PixelTrackSoAFromCUDA::PixelTrackSoAFromCUDA(const edm::ParameterSet& iConfig)
    //: tokenCUDA_(consumes<cms::cuda::Product<PixelTrackHeterogeneous>>(iConfig.getParameter<edm::InputTag>("src"))),
    //  tokenSOA_(produces<PixelTrackHeterogeneous>()) {}
    : tokenCUDA_(consumes<cms::cuda::Product<pixelTrack::TrackSoA>>(iConfig.getParameter<edm::InputTag>("src"))),
      //tokenSOA_(produces<TrackSoAHeterogeneousT_test<>::View>()) {}
      tokenSOA_(produces<pixelTrack::TrackSoA>()) {}

void PixelTrackSoAFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("pixelTracksCUDA"));
  descriptions.add("pixelTracksSoA", desc);
}

/*void PixelTrackSoAFromCUDA::acquire(edm::Event const& iEvent,
                                    edm::EventSetup const& iSetup,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<PixelTrackHeterogeneous> const& inputDataWrapped = iEvent.get(tokenCUDA_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  soa_ = inputData.toHostAsync(ctx.stream());
}*/

void PixelTrackSoAFromCUDA::acquire(edm::Event const& iEvent,
                                    edm::EventSetup const& iSetup,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<pixelTrack::TrackSoA> const& inputDataWrapped = iEvent.get(tokenCUDA_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  //class_ = inputData.toHostAsync(ctx.stream());

  pixelTrack::TrackSoA soa_(ctx.stream());
  cudaCheck(cudaMemcpy(&soa_,&inputData,sizeof(pixelTrack::TrackSoA),cudaMemcpyDeviceToHost));

  auto retView = cms::cuda::make_host_unique<std::byte[]>(inputData.bufferSize(), ctx.stream());
  cudaCheck(cudaMemcpy(retView.get(),inputData.buffer().get(),TrackSoAHeterogeneousT_test<>::computeDataSize(32768),cudaMemcpyDeviceToHost));
  TrackSoAHeterogeneousT_test<> tmp_layout(retView.get(),32768);
  TrackSoAHeterogeneousT_test<>::View tmp_view_(tmp_layout);

}

void PixelTrackSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // check that the fixed-size SoA does not overflow
  //auto tsoa = soa_;
  //auto maxTracks = tsoa.stride();
  auto maxTracks = 32768;
  auto nTracks = tmp_view_.nTracks();
  assert(nTracks < maxTracks);
  if (nTracks == maxTracks - 1) {
    edm::LogWarning("PixelTracks") << "Unsorted reconstructed pixel tracks truncated to " << maxTracks - 1
                                   << " candidates";
  }

#ifdef PIXEL_DEBUG_PRODUCE
  std::cout << "size of SoA " << sizeof(soa_) << " stride " << maxTracks << std::endl;
  std::cout << "found " << nTracks << " tracks in cpu SoA at " << &soa_ << std::endl;

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = soa_.nHits(it);
    assert(nHits == int(soa_.hitIndices.size(it)));
    if (nHits == 0)
      break;  // this is a guard: maybe we need to move to nTracks...
    nt++;
  }
  assert(nTracks == nt);
#endif

  // DO NOT  make a copy  (actually TWO....)
  iEvent.emplace(tokenSOA_, std::move(soa_));//, std::move(ret)); // view

  //assert(!soa_);
}

DEFINE_FWK_MODULE(PixelTrackSoAFromCUDA);
