#ifndef DataFormats_BeamSpot_interface_alpaka_BeamSpotDevice_h
#define DataFormats_BeamSpot_interface_alpaka_BeamSpotDevice_h

#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "DataFormats/Portable/interface/alpaka/PortableObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // simplified representation of the beamspot data, in device global memory
  using BeamSpotDevice = PortableObject<BeamSpotPOD>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_BeamSpot_interface_alpaka_BeamSpotDevice_h
