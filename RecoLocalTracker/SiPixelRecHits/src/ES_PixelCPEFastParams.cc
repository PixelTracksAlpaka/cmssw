#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParamsHost.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

// Compilation of this file informs framework about the production of the CPEFastParamsHost ESProduct through type lookup data

using PixelCPEFastParamsHostPhase1 = PixelCPEFastParamsHost<pixelTopology::Phase1>;
using PixelCPEFastParamsHostPhase2 = PixelCPEFastParamsHost<pixelTopology::Phase2>;

TYPELOOKUP_DATA_REG(PixelCPEFastParamsHostPhase1);
TYPELOOKUP_DATA_REG(PixelCPEFastParamsHostPhase2);
