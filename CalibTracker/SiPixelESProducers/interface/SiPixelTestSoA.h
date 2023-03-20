#ifndef CalibTracker_SiPixelES_interface_PixelESTestSoA_h
#define CalibTracker_SiPixelES_interface_PixelESTestSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

GENERATE_SOA_LAYOUT(PixelESTestSoALayout, SOA_COLUMN(int, z))

using PixelESTestSoA = PixelESTestSoALayout<>;

#endif
