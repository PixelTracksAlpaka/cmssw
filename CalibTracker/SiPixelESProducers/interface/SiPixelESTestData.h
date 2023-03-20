#ifndef CalibTracker_SiPixelES_interface_PixelESTestData_h
#define CalibTracker_SiPixelES_interface_PixelESTestData_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelTestSoA.h"

using PixelESTestDataHost = PortableHostCollection<PixelESTestSoA>;

#endif
