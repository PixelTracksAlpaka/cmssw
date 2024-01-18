#ifndef RecoPixelVertexing_PixelVertexFinding_PixelVertexWorkSpaceSoADevice_h
#define RecoPixelVertexing_PixelVertexFinding_PixelVertexWorkSpaceSoADevice_h
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/Vertex1DSoA/interface/alpaka/Vertex1DUtilities.h"
#include "DataFormats/Vertex1DSoA/interface/Vertex1DDefinitions.h"
#include "../PixelVertexWorkSpaceLayout.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <int32_t S>
  class PixelVertexWorkSpaceSoADevice : public PortableCollection<PixelVertexWSSoALayout<>> {
  public:
    PixelVertexWorkSpaceSoADevice() = default;

    // Constructor which specifies the SoA size and Alpaka Queue
    explicit PixelVertexWorkSpaceSoADevice(Queue queue) : PortableCollection<PixelVertexWSSoALayout<>>(S, queue) {}

    // Constructor which specifies the SoA size and alpaka device
    // TODO: Needed?
    explicit PixelVertexWorkSpaceSoADevice(Device const& device)
        : PortableCollection<PixelVertexWSSoALayout<>>(S, device) {}
  };
  namespace vertexFinder {
    namespace workSpace {
      using PixelVertexWorkSpaceSoADevice = PixelVertexWorkSpaceSoADevice<::vertex1d::MAXTRACKS>;
    }  // namespace workSpace
  }    // namespace vertexFinder
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
