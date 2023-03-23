#ifndef CalibTracker_SiPixelESProducers_SiPixelGainCalibrationForHLTDevice_h
#define CalibTracker_SiPixelESProducers_SiPixelGainCalibrationForHLTDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTLayout.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using SiPixelGainCalibrationForHLTDevice = PortableCollection<SiPixelGainCalibrationForHLTLayout<>>;
  using SiPixelGainCalibrationForHLTHost = PortableHostCollection<SiPixelGainCalibrationForHLTLayout<>>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif  // CalibTracker_SiPixelESProducers_SiPixelGainCalibrationForHLTDevice_h
