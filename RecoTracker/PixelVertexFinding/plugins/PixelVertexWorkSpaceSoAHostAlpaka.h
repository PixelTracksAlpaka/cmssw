#ifndef RecoPixelVertexing_PixelVertexFinding_PixelVertexWorkSpaceSoAHost_h
#define RecoPixelVertexing_PixelVertexFinding_PixelVertexWorkSpaceSoAHost_h
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Vertex/interface/alpaka/ZVertexUtilities.h"
#include "RecoTracker/PixelVertexFinding/interface/PixelVertexWorkSpaceLayout.h"

namespace vertexFinder {
    using PixelVertexWorkSpaceSoAHost = PortableHostCollection<PixelVertexWSSoALayout<>>;//(zVertex::MAXTRACKS, queue) {};//PixelVertexWorkSpaceSoAHost<zVertex::MAXTRACKS>;
}  // namespace vertexFinder
#endif
