#ifndef CUDADataFormats_RecHits_TrackingRecHitsDevice_h
#define CUDADataFormats_RecHits_TrackingRecHitsDevice_h

#include <cstdint>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

template<typename TrackerTraits>
class TrackingRecHitSoADevice : public cms::cuda::PortableDeviceCollection<HitLayout<TrackerTraits>> {
public:

  using hitSoA = trackingRecHitSoA<TrackerTraits>;
  //Need to decorate the class with the inherited portable accessors being now a template
  using cms::cuda::PortableDeviceCollection<HitLayout<TrackerTraits>>::view;
  using cms::cuda::PortableDeviceCollection<HitLayout<TrackerTraits>>::const_view;
  using cms::cuda::PortableDeviceCollection<HitLayout<TrackerTraits>>::buffer;
  using cms::cuda::PortableDeviceCollection<HitLayout<TrackerTraits>>::bufferSize;

  TrackingRecHitSoADevice() = default;  // cms::cuda::Product needs this

  using AverageGeometry = typename hitSoA::AverageGeometry;;
  using ParamsOnGPU = typename hitSoA::ParamsOnGPU;;
  using PhiBinnerStorageType = typename hitSoA::PhiBinnerStorageType;
  using PhiBinner = typename hitSoA::PhiBinner;
  // Constructor which specifies the SoA size
  explicit TrackingRecHitSoADevice(uint32_t nHits,
                                   int32_t offsetBPIX2,
                                   ParamsOnGPU const* cpeParams,
                                   uint32_t const* hitsModuleStart,
                                   cudaStream_t stream)
      : cms::cuda::PortableDeviceCollection<HitLayout<TrackerTraits>>(nHits, stream),
        nHits_(nHits),
        cpeParams_(cpeParams),
        hitsModuleStart_(hitsModuleStart),
        offsetBPIX2_(offsetBPIX2) {
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    phiBinner_ = &(view().phiBinner());
    // phiBinner_ = cms::cuda::make_device_unique<TrackingRecHit2DSOAView::PhiBinner>(stream).get();
    cudaCheck(cudaMemcpyAsync(&(view().nHits()), &nHits, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(view().hitsModuleStart().data(),
                              hitsModuleStart,
                              sizeof(uint32_t) * int(TrackerTraits::numberOfModules + 1),
                              cudaMemcpyHostToDevice,
                              stream));
    cudaCheck(
        cudaMemcpyAsync(&(view().offsetBPIX2()), &offsetBPIX2, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    // cpeParams argument is a pointer to device memory, copy
    // its contents into the Layout.

    cudaCheck(cudaMemcpyAsync(
        &(view().cpeParams()), cpeParams, int(sizeof(ParamsOnGPU)), cudaMemcpyDeviceToDevice, stream));
  }

  uint32_t nHits() const { return nHits_; }  //go to size of view
  // uint32_t nModules() const { return nModules_; }

  cms::cuda::host::unique_ptr<float[]> localCoordToHostAsync(cudaStream_t stream) const {
    auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), stream);
    size_t rowSize = sizeof(float) * nHits();
    // printf("nModules=%d \n", nModules());
    // printf("nHits=%d \n", nHits());
    cudaCheck(cudaMemcpyAsync(ret.get(), view().xLocal(), rowSize * 4, cudaMemcpyDeviceToHost, stream));
    // cudaCheck(cudaMemcpyAsync(ret.get() + rowSize , view().yLocal() , rowSize, cudaMemcpyDeviceToHost, stream));
    // cudaCheck(cudaMemcpyAsync(ret.get() + size_t(rowSize * 2), view().xerrLocal() , rowSize, cudaMemcpyDeviceToHost, stream));
    // cudaCheck(cudaMemcpyAsync(ret.get() + size_t(rowSize * 3) , view().yerrLocal() , rowSize, cudaMemcpyDeviceToHost, stream));
    return ret;
  }  //move to utilities

  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const {
    // printf("%d \n",nModules());
    auto ret = cms::cuda::make_host_unique<uint32_t[]>(TrackerTraits::numberOfModules + 1, stream);
    cudaCheck(cudaMemcpyAsync(ret.get(),
                              view().hitsModuleStart().data(),
                              sizeof(uint32_t) * (TrackerTraits::numberOfModules + 1),
                              cudaMemcpyDeviceToHost,
                              stream));
    return ret;
  }

  auto phiBinnerStorage() { return phiBinnerStorage_; }
  auto hitsModuleStart() const { return hitsModuleStart_; }
  uint32_t offsetBPIX2() const { return offsetBPIX2_; }
  auto phiBinner() { return phiBinner_; }


private:
  uint32_t nHits_;  //Needed for the host SoA size
  ParamsOnGPU const* cpeParams_;  //TODO: this is used not that much from the hits (only once in BrokenLineFit), would make sens to remove it from this class.
  uint32_t const* hitsModuleStart_;
  uint32_t offsetBPIX2_;

  PhiBinnerStorageType* phiBinnerStorage_;
  PhiBinner* phiBinner_;

};


//Classes definition for Phase1/Phase2, to make the classes_def lighter. Not actually used in the code.
using TrackingRecHitSoADevicePhase1 = TrackingRecHitSoADevice<pixelTopology::Phase1>;
using TrackingRecHitSoADevicePhase2 = TrackingRecHitSoADevice<pixelTopology::Phase2>;


#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
