#include "FWCore/Utilities/interface/typelookup.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParamsDevice.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

//Compilation of this file informs framework about the production of the CPEFastParams ESProduct through type lookup data

using PixelCPEFastParamsPhase1 = PixelCPEFastParamsDevice<alpaka_common::DevHost, pixelTopology::Phase1>;
using PixelCPEFastParamsPhase2 = PixelCPEFastParamsDevice<alpaka_common::DevHost, pixelTopology::Phase2>;

TYPELOOKUP_DATA_REG(PixelCPEFastParamsPhase1);
TYPELOOKUP_DATA_REG(PixelCPEFastParamsPhase2);