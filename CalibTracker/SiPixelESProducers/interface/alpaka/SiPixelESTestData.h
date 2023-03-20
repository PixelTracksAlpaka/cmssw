#ifndef CalibTracker_SiPixelES_interface_alpaka_PixelESTestData_h
#define CalibTracker_SiPixelES_interface_alpaka_PixelESTestData_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelTestSoA.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelESTestData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using PixelESTestDataHost = PixelESTestDataHost;
  using PixelESTestDataDevice = PortableCollection<PixelESTestSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
