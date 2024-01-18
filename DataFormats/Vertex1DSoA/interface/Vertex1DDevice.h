#ifndef DataFormats_Vertex1DSoA_interface_Vertex1DDevice_h
#define DataFormats_Vertex1DSoA_interface_Vertex1DDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>
#include "DataFormats/Vertex1DSoA/interface/Vertex1DSoA.h"
#include "DataFormats/Vertex1DSoA/interface/Vertex1DDefinitions.h"
#include "DataFormats/Vertex1DSoA/interface/alpaka/Vertex1DUtilities.h"
#include "DataFormats/Vertex1DSoA/interface/Vertex1DHost.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

template <int32_t S, typename TDev>
class Vertex1DDeviceSoA : public PortableDeviceCollection<Vertex1DLayout<>, TDev> {
public:
  Vertex1DDeviceSoA() = default;  // necessary for ROOT dictionaries

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit Vertex1DDeviceSoA(TQueue queue) : PortableDeviceCollection<Vertex1DLayout<>, TDev>(S, queue) {}
};

using namespace ::vertex1d;
template <typename TDev>
using Vertex1DDevice = Vertex1DDeviceSoA<MAXTRACKS, TDev>;

#endif  // DataFormats_Vertex1DSoA_interface_Vertex1DDevice_h
