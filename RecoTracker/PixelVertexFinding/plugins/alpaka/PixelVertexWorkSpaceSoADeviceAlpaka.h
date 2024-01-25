#ifndef RecoPixelVertexing_PixelVertexFinding_PixelVertexWorkSpaceSoADevice_h
#define RecoPixelVertexing_PixelVertexFinding_PixelVertexWorkSpaceSoADevice_h
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "DataFormats/VertexSoA/interface/ZVertexDefinitions.h"
#include "RecoTracker/PixelVertexFinding/interface/PixelVertexWorkSpaceLayout.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  
  namespace vertexFinder {
      
      using PixelVertexWorkSpaceSoADevice = PortableCollection<::vertexFinder::PixelVertexWSSoALayout<>>;
  }    // namespace vertexFinder
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
