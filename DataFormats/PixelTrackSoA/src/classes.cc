#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/PixelTrackSoA/interface/PixelTrackSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<PixelTrackLayout<pixelTopology::Phase1>>);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<PixelTrackLayout<pixelTopology::Phase2>>);
// SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<PixelTrackLayout<pixelTopology::HIonPhase1>>);