#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/TrackerRecHitSoA/interface/TrackerRecHitSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackerRecHitLayout<pixelTopology::Phase1>>);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackerRecHitLayout<pixelTopology::Phase2>>);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackerRecHitLayout<pixelTopology::HIonPhase1>>);