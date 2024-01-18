#ifndef DataFormats_Vertex1DSoA_interface_Vertex1DHost_h
#define DataFormats_Vertex1DSoA_interface_Vertex1DHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Vertex1DSoA/interface/Vertex1DSoA.h"
#include "DataFormats/Vertex1DSoA/interface/Vertex1DDefinitions.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

template <int32_t S>
class Vertex1DHostSoA : public PortableHostCollection<Vertex1DLayout<>> {
public:
  Vertex1DHostSoA() = default;

  // Constructor which specifies the queue
  template <typename TQueue>
  explicit Vertex1DHostSoA(TQueue queue) : PortableHostCollection<Vertex1DLayout<>>(S, queue) {}

  // Constructor which specifies the DevHost
  explicit Vertex1DHostSoA(alpaka_common::DevHost const& host) : PortableHostCollection<Vertex1DLayout<>>(S, host) {}
};

using namespace ::vertex1d;
using Vertex1DHost = Vertex1DHostSoA<MAXTRACKS>;

#endif  // DataFormats_Vertex1DSoA_interface_Vertex1DHost_h
