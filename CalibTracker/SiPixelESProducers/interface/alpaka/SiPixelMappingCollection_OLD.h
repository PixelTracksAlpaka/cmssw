#ifndef CalibTracker_SiPixelESProducers_interface_alpaka_SiPixelMappingCollection_h
#define CalibTracker_SiPixelESProducers_interface_alpaka_SiPixelMappingCollection_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelMappingLayout.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelMappingHost.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelMappingDevice.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  using SiPixelMappingCollection = SiPixelMappingHost;
#else
  using SiPixelMappingCollection = SiPixelMappingDevice<Device>;
#endif

  using SiPixelMappingSoA = SiPixelMappingCollection;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// namespace cms::alpakatools {
//   template <>
//   struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelMappingSoA> {
//     template <typename TQueue>
//     static auto copyAsync(TQueue& queue, ALPAKA_ACCELERATOR_NAMESPACE::SiPixelMappingSoA const& deviceData) {
//       SiPixelMappingHost hostData(queue);
//       alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
//       return hostData;
//     }
//   };
// }  // namespace cms::alpakatools

#endif  // CalibTracker_SiPixelESProducers_interface_alpaka_SiPixelMappingCollection_h
