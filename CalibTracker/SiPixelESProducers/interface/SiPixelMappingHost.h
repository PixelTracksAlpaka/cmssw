#ifndef CalibTracker_SiPixelESProducers_interface_SiPixelMappingHost_h
#define CalibTracker_SiPixelESProducers_interface_SiPixelMappingHost_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/PortableHostCollection.h"
// #include "CalibTracker/SiPixelESProducers/interface/alpaka/SiPixelMappingDevice.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelMappingLayout.h"

using SiPixelMappingHost = PortableHostCollection<SiPixelMappingLayout<>>;

// namespace cms::alpakatools {
//   template <>
//   struct CopyToDevice<SiPixelMappingHost> {
//     template <typename TQueue>
//     static auto copyAsync(TQueue& queue, SiPixelMappingHost const& srcData) {
//       //auto dstBuffer = cms::alpakatools::make_device_buffer<int[]>(queue, src.size());
//       SiPixelMappingDevice<alpaka::Dev<TQueue>> devData;
//       alpaka::memcpy(queue, devData.buffer(), srcData.buffer());
//       //   return SiPixelMappingDevice<alpaka::Dev<TQueue>>(std::move(dstBuffer));
//       return devData;
//     }
//   };
// }  // namespace cms::alpakatools

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

#endif  // CalibTracker_SiPixelESProducers_interface_SiPixelMappingHost_h
