#ifndef CUDADataFormats_Track_TrackHeterogeneousHost_H
#define CUDADataFormats_Track_TrackHeterogeneousHost_H

#include <cstdint>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

class TrackingRecHitSoAHost : public cms::cuda::PortableHostCollection<TrackingRecHitSoALayout<>> {
public:
  TrackingRecHitSoAHost() = default;

  // This SoA Host is used basically only for DQM
  // so we  just need a slim constructor
  explicit TrackingRecHitSoAHost(uint32_t nHits, cudaStream_t stream)
  : PortableHostCollection<TrackingRecHitSoALayout<>>(nHits, stream) {}

  explicit TrackingRecHitSoAHost(uint32_t nHits, bool isPhase2, int32_t offsetBPIX2, pixelCPEforGPU::ParamsOnGPU const* cpeParams, uint32_t const* hitsModuleStart, cudaStream_t stream)
      : PortableHostCollection<TrackingRecHitSoALayout<>>(nHits, stream), nHits_(nHits), cpeParams_(cpeParams)
      {
        nModules_ = isPhase2 ? phase2PixelTopology::numberOfModules : phase1PixelTopology::numberOfModules;

        view().nHits() = nHits;
        view().nMaxModules() = nModules_;
        std::copy(hitsModuleStart,hitsModuleStart+nModules_,view().hitsModuleStart().begin());

        view().offsetBPIX2() = offsetBPIX2;

      }

  uint32_t nHits() const { return nHits_; }
  uint32_t nModules() const { return nModules_; }
  auto phiBinnerStorage() { return phiBinnerStorage_; }

  private:
    uint32_t nHits_; //Needed for the host SoA size
    pixelCPEforGPU::ParamsOnGPU const* cpeParams_;
    uint32_t nModules_;
    trackingRecHitSoA::PhiBinnerStorageType* phiBinnerStorage_;
};


#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
