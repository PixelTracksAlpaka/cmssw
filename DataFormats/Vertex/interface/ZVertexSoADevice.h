#ifndef DataFormats_Vertex_ZVertexSoADevice_H
#define DataFormats_Vertex_ZVertexSoADevice_H

#include <cstdint>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/Vertex/interface/ZVertexDefinitions.h"
#include "DataFormats/Vertex/interface/ZVertexLayout.h"
#include "DataFormats/Vertex/interface/ZVertexSoAHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <int32_t S, typename TDev>
class ZVertexSoADevice : public PortableDeviceCollection<ZVertexSoAHeterogeneousLayout<>, TDev> {
public:
  ZVertexSoADevice() = default;  // cms::alpakatools::Product needs this

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit ZVertexSoADevice(TQueue queue) : PortableDeviceCollection<ZVertexSoAHeterogeneousLayout<>, TDev>(S, queue) {}

  // Constructor which specifies the SoA size
  explicit ZVertexSoADevice(TDev const& device)
      : PortableDeviceCollection<ZVertexSoAHeterogeneousLayout<>, TDev>(S, device) {}
};

using namespace ::zVertex;
template <typename TDev>
using ZVertexDevice = ZVertexSoADevice<MAXTRACKS, TDev>;

#endif  // DataFormats_Vertex_ZVertexSoADevice_H