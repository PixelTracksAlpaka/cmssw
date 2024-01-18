#ifndef DataFormats_Vertex1DSoA_interface_alpaka_Vertex1DSoACollection_h
#define DataFormats_Vertex1DSoA_interface_alpaka_Vertex1DSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/Vertex1DSoA/interface/Vertex1DSoA.h"
#include "DataFormats/Vertex1DSoA/interface/Vertex1DDefinitions.h"
#include "DataFormats/Vertex1DSoA/interface/alpaka/Vertex1DUtilities.h"
#include "DataFormats/Vertex1DSoA/interface/Vertex1DHost.h"
#include "DataFormats/Vertex1DSoA/interface/Vertex1DDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using Vertex1DCollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, Vertex1DHost, Vertex1DDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TDevice>
  struct CopyToHost<Vertex1DDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, Vertex1DDevice<TDevice> const& deviceData) {
      Vertex1DHost hostData(queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
#ifdef GPU_DEBUG
      printf("Vertex1DCollection: I'm copying to host.\n");
#endif
      return hostData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(Vertex1DCollection, Vertex1DHost);

#endif  // DataFormats_Vertex1DSoA_interface_alpaka_Vertex1DSoACollection_h
