#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "DataFormats/Portable/interface/PortableCUDADeviceCollection.h"

// Host and device layout: data used on both sides and transferred from device to host.
GENERATE_SOA_LAYOUT(SiPixelDigisCUDATemplate,
                    SOA_COLUMN(int32_t, clus),
                    SOA_COLUMN(uint32_t, pdigi),
                    SOA_COLUMN(uint32_t, rawIdArr),
                    SOA_COLUMN(uint16_t, adc),
                    SOA_COLUMN(uint16_t, xx),
                    SOA_COLUMN(uint16_t, yy),
                    SOA_COLUMN(uint16_t, moduleId))


  using SiPixelDigisCUDALayout = SiPixelDigisCUDATemplate<>;
  using SiPixelDigisCUDAView = SiPixelDigisCUDALayout::View;
  using SiPixelDigisCUDAConstView = SiPixelDigisCUDALayout::ConstView;

// While porting from previous code, we decorate the base PortableCollection. XXX/TODO: improve if possible...
class SiPixelDigisCUDA : public PortableCUDADeviceCollection<SiPixelDigisCUDALayout> {
public:

  SiPixelDigisCUDA() = default;
  explicit SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream) :
      PortableCUDADeviceCollection<SiPixelDigisCUDALayout> (maxFedWords + 1, stream)
      {
        assert(maxFedWords != 0);

      }



  // movable
  SiPixelDigisCUDA(SiPixelDigisCUDA &&) = default;
  SiPixelDigisCUDA &operator=(SiPixelDigisCUDA &&) = default;

  void setNModulesDigis(uint32_t nModules, uint32_t nDigis) {
    nModules_h = nModules;
    nDigis_h = nDigis;
  }

  ~SiPixelDigisCUDA() = default;

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return nDigis_h; }

  cms::cuda::host::unique_ptr<std::byte[]> copyAllToHostAsync(cudaStream_t stream) const {
    // Copy to a host buffer the host-device shared part (m_hostDeviceLayout).
    auto ret = cms::cuda::make_host_unique<std::byte[]>(layout().metadata().byteSize(), stream);
    cudaCheck(cudaMemcpyAsync(
        ret.get(), layout().metadata().data(), layout().metadata().byteSize(), cudaMemcpyDeviceToHost, stream));
    return ret;
  }

private:
  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif  // CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
