#ifndef DataFormats_SiPixelGainCalibrationForHLTSoA_SiPixelGainCalibrationForHLTHost_h

#define DataFormats_SiPixelGainCalibrationForHLTSoA_SiPixelGainCalibrationForHLTHost_h


#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "SiPixelGainCalibrationForHLTLayout.h"

using SiPixelGainCalibrationForHLTHost = PortableHostCollection<SiPixelGainCalibrationForHLTLayout<>>;

#endif  // DataFormats_SiPixelGainCalibrationForHLTSoA_SiPixelGainCalibrationForHLTHost_h

