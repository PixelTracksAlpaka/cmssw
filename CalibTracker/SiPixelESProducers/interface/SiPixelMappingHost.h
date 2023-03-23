#ifndef CalibTracker_SiPixelESProducers_interface_SiPixelClustersDevice_h
#define CalibTracker_SiPixelESProducers_interface_SiPixelClustersDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "SiPixelMappingLayout.h"

using SiPixelMappingHost = PortableHostCollection<SiPixelMappingLayout<>>;

// class SiPixelFedCablingMap;

// class SiPixelMappingHost : public PortableHostCollection<SiPixelMappingLayout<>> {
// public:
//   SiPixelMappingHost() = default;
//   ~SiPixelMappingHost() = default;

//   template <typename TQueue>
//   explicit SiPixelMappingHost(size_t maxModules, SiPixelFedCablingMap const& cablingMap, bool hasQuality, TQueue queue)
//       : PortableHostCollection<SiPixelMappingLayout<>>(maxModules + 1, queue), hasQuality_(hasQuality),
//        cablingMap_(&cablingMap)
//        {}

//   SiPixelMappingHost(SiPixelMappingHost &&) = default;
//   SiPixelMappingHost &operator=(SiPixelMappingHost &&) = default;

//   bool hasQuality() const { return hasQuality_; }

// private:
//   bool hasQuality_;
//   const SiPixelFedCablingMap *cablingMap_; //this is the cabling map that is ALWAYS on Host
// };

#endif  // CalibTracker_SiPixelESProducers_interface_SiPixelClustersDevice_h
