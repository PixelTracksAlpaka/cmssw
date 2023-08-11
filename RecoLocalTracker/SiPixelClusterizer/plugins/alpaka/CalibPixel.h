#ifndef RecoLocalTracker_SiPixelClusterizer_alpaka_CalibPixel_h
#define RecoLocalTracker_SiPixelClusterizer_alpaka_CalibPixel_h

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLTLayout.h"
#include "CondFormats/SiPixelObjects/interface/alpaka/SiPixelGainCalibrationForHLTDevice.h"
#include "CondFormats/SiPixelObjects/interface/alpaka/SiPixelGainCalibrationForHLTUtilities.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoAv2.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelClusterThresholds.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace calibPixel {
    using namespace cms::alpakatools;
    constexpr uint16_t InvId = std::numeric_limits<uint16_t>::max() - 1;
    // must be > MaxNumModules

    // valid for run2
    constexpr float VCaltoElectronGain = 47;         // L2-4: 47 +- 4.7
    constexpr float VCaltoElectronGain_L1 = 50;      // L1:   49.6 +- 2.6
    constexpr float VCaltoElectronOffset = -60;      // L2-4: -60 +- 130
    constexpr float VCaltoElectronOffset_L1 = -670;  // L1:   -670 +- 220

    //for phase2
    constexpr float ElectronPerADCGain = 1500;
    constexpr int8_t Phase2ReadoutMode = 3;
    constexpr uint16_t Phase2DigiBaseline = 1000;
    constexpr uint8_t Phase2KinkADC = 8;

    struct calibDigis {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    SiPixelClusterThresholds clusterThresholds,
                                    SiPixelDigisSoAv2View view,
                                    SiPixelClustersSoAView clus_view,
                                    const SiPixelGainCalibrationForHLTSoAConstView gains,
                                    int numElements) const {
        const uint32_t threadIdxGlobal(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);

        const float VCaltoElectronGain = clusterThresholds.vCaltoElectronGain;
        const float VCaltoElectronGain_L1 = clusterThresholds.vCaltoElectronGain_L1;
        const float VCaltoElectronOffset = clusterThresholds.vCaltoElectronOffset;
        const float VCaltoElectronOffset_L1 = clusterThresholds.vCaltoElectronOffset_L1;


        // zero for next kernels...
        if (threadIdxGlobal == 0) {
          clus_view[0].clusModuleStart() = clus_view[0].moduleStart() = 0;
        }

        cms::alpakatools::for_each_element_in_grid_strided(
            acc, phase1PixelTopology::numberOfModules, [&](uint32_t i) { clus_view[i].clusInModule() = 0; });

        cms::alpakatools::for_each_element_in_grid_strided(acc, numElements, [&](uint32_t i) {
          auto dvgi = view[i];
          if (dvgi.moduleId() != InvId) {
        
            bool isDeadColumn = false, isNoisyColumn = false;

            int row = dvgi.xx();
            int col = dvgi.yy();

            auto ret = SiPixelGainUtilities::getPedAndGain(gains, dvgi.moduleId(), col, row, isDeadColumn, isNoisyColumn);

            float pedestal = ret.first;
            float gain = ret.second;
            if (isDeadColumn | isNoisyColumn) {
              dvgi.moduleId() = InvId;
              dvgi.adc() = 0;
              printf("bad pixel at %d in %d\n", i, dvgi.moduleId());
            } else {
              float vcal = dvgi.adc() * gain - pedestal * gain;
              
              float conversionFactor = dvgi.moduleId() < 96 ? VCaltoElectronGain_L1 : VCaltoElectronGain;
              float offset = dvgi.moduleId() < 96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset;
              
              vcal = vcal * conversionFactor + offset;

              dvgi.adc() = std::max(100, int(vcal * conversionFactor + offset));
            }
          }
        });
      }
    };

    struct calibDigisPhase2 {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    SiPixelClusterThresholds clusterThresholds,
                                    SiPixelDigisSoAv2View& view,
                                    SiPixelClustersSoAView& clus_view,
                                    int numElements
      ) const {

        const float ElectronPerADCGain = clusterThresholds.electronPerADCGain;
        const int8_t Phase2ReadoutMode = clusterThresholds.phase2ReadoutMode;
        const uint16_t Phase2DigiBaseline = clusterThresholds.phase2DigiBaseline;
        const uint8_t Phase2KinkADC = clusterThresholds.phase2KinkADC;
        
        const uint32_t threadIdxGlobal(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        // zero for next kernels...

        if (0 == threadIdxGlobal)
          clus_view[0].clusModuleStart() = clus_view[0].moduleStart();
        cms::alpakatools::for_each_element_in_grid_strided(
            acc, phase2PixelTopology::numberOfModules, [&](uint32_t i) { clus_view[i].clusInModule() = 0; });

        cms::alpakatools::for_each_element_in_grid_strided(acc, numElements, [&](uint32_t i) {
          auto dvgi = view[i];
          if (pixelClustering::invalidModuleId != dvgi.moduleId()) {
            const int mode = (Phase2ReadoutMode < -1 ? -1 : Phase2ReadoutMode);

            int adc_int = dvgi.adc();

            if (mode < 0)
              adc_int = int(adc_int * ElectronPerADCGain);
            else {
              if (adc_int < Phase2KinkADC)
                adc_int = int((adc_int + 0.5) * ElectronPerADCGain);
              else {
                const int8_t dspp = (Phase2ReadoutMode < 10 ? Phase2ReadoutMode : 10);
                const int8_t ds = int8_t(dspp <= 1 ? 1 : (dspp - 1) * (dspp - 1));

                adc_int -= Phase2KinkADC;
                adc_int *= ds;
                adc_int += Phase2KinkADC;

                adc_int = ((adc_int + 0.5 * ds) * ElectronPerADCGain);
              }

              adc_int += int(Phase2DigiBaseline);
            }
            dvgi.adc() = std::min(adc_int, int(std::numeric_limits<uint16_t>::max()));
          }
        });
      }
    };

  }  // namespace calibPixel
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelClusterizer_alpaka_CalibPixel.h
