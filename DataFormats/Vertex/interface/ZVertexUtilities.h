#ifndef DataFormats_Vertex_ZVertexUtilities_h
#define DataFormats_Vertex_ZVertexUtilities_h

#include "DataFormats/Vertex/interface/ZVertexLayout.h"

// Previous ZVertexSoA class methods.
// They operate on View and ConstView of the ZVertexSoA.
namespace zVertex {

  constexpr uint32_t MAXTRACKS = 32 * 1024; //TODO: make me dependend on TrackerTraits.
  constexpr uint32_t MAXVTX = 1024;         //TODO: also me, thanks.

  namespace utilities {
    using ZVertexSoALayout = ZVertexSoAHeterogeneousLayout<>;
    using ZVertexSoAView = ZVertexSoAHeterogeneousLayout<>::View;
    using ZVertexSoAConstView = ZVertexSoAHeterogeneousLayout<>::ConstView;

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void init(ZVertexSoAView &vertices) { vertices.nvFinal() = 0; }

  }  // namespace utilities
}    // namespace zVertex

#endif
