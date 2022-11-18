#ifndef CUDADataFormats_Track_TrackHeterogeneousDevice_H
#define CUDADataFormats_Track_TrackHeterogeneousDevice_H

#include <cstdint>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

class TrackingRecHitSoADevice : public cms::cuda::PortableDeviceCollection<TrackingRecHitSoALayout<>> {
public:
  TrackingRecHitSoADevice() = default;  // cms::cuda::Product needs this

  // Constructor which specifies the SoA size
  explicit TrackingRecHitSoADevice(uint32_t nHits, bool isPhase2, int32_t offsetBPIX2, pixelCPEforGPU::ParamsOnGPU const* cpeParams, uint32_t const* hitsModuleStart, cudaStream_t stream)
      : PortableDeviceCollection<TrackingRecHitSoALayout<>>(nHits, stream), nHits_(nHits), cpeParams_(cpeParams), hitsModuleStart_(hitsModuleStart)
      {
        nModules_ = isPhase2 ? phase2PixelTopology::numberOfModules : phase1PixelTopology::numberOfModules;
        phiBinner_ = &(view().phiBinner());
        // phiBinner_ = cms::cuda::make_device_unique<TrackingRecHit2DSOAView::PhiBinner>(stream).get();
        cudaCheck(cudaMemcpyAsync(&(view().nHits()), &nHits, sizeof(uint32_t),cudaMemcpyHostToDevice,stream));
        cudaCheck(cudaMemcpyAsync(&(view().nMaxModules()), &nModules_, sizeof(uint32_t),cudaMemcpyHostToDevice,stream));
        cudaCheck(cudaMemcpyAsync(&(view().hitsModuleStart()), hitsModuleStart, sizeof(uint32_t) * int(nModules_),cudaMemcpyHostToDevice,stream));
        // cudaCheck(cudaMemcpyAsync(&(view().cpeParams()), cpeParams, int(sizeof(pixelCPEforGPU::ParamsOnGPU)),cudaMemcpyHostToDevice,stream));
        cudaCheck(cudaMemcpyAsync(&(view().offsetBPIX2()), &offsetBPIX2, sizeof(int32_t),cudaMemcpyHostToDevice,stream));

      }

  uint32_t nHits() const { return nHits_; } //go to size of view
  uint32_t nModules() const { return nModules_; }

  cms::cuda::host::unique_ptr<float[]> localCoordToHostAsync(cudaStream_t stream) const {
    auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), stream);
    size_t rowSize = sizeof(float) * nHits();
    printf("%d \n",nModules());
    printf("%d \n",nHits());
    cudaCheck(cudaMemcpyAsync(ret.get(), view().xLocal() , rowSize * 4, cudaMemcpyDeviceToHost, stream));
    // cudaCheck(cudaMemcpyAsync(ret.get() + rowSize , view().yLocal() , rowSize, cudaMemcpyDeviceToHost, stream));
    // cudaCheck(cudaMemcpyAsync(ret.get() + size_t(rowSize * 2), view().xerrLocal() , rowSize, cudaMemcpyDeviceToHost, stream));
    // cudaCheck(cudaMemcpyAsync(ret.get() + size_t(rowSize * 3) , view().yerrLocal() , rowSize, cudaMemcpyDeviceToHost, stream));
    return ret;
  } //move to utilities


  auto phiBinnerStorage() { return phiBinnerStorage_; }
  auto hitsModuleStart() const { return hitsModuleStart_; }
  auto phiBinner() { return phiBinner_; }

  private:
    uint32_t nHits_; //Needed for the host SoA size
    pixelCPEforGPU::ParamsOnGPU const* cpeParams_; //TODO: this is used not that much from the hits (only once in BrokenLineFit), would make sens to remove it from this class.
    uint32_t const* hitsModuleStart_;

    uint32_t nModules_;
    trackingRecHitSoA::PhiBinnerStorageType* phiBinnerStorage_;
    trackingRecHitSoA::PhiBinner* phiBinner_;

};


#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
