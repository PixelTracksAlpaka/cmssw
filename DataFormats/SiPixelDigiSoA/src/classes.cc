#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/Layout.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<SiPixelDigiErrorsSoA>);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<Layout>);