#ifndef DataFormats_Vertex1DSoA_interface_alpaka_Vertex1DUtilities_h
#define DataFormats_Vertex1DSoA_interface_alpaka_Vertex1DUtilities_h
#include <alpaka/alpaka.hpp>
#include "DataFormats/Vertex1DSoA/interface/Vertex1DSoA.h"
#include "DataFormats/Vertex1DSoA/interface/Vertex1DDefinitions.h"

// Previous Vertex1DSoA class methods.
// They operate on View and ConstView of the Vertex1DSoA.
namespace vertex1d {
  namespace utilities {
    using Vertex1DSoA = Vertex1DLayout<>;
    using Vertex1DSoAView = Vertex1DLayout<>::View;
    using Vertex1DSoAConstView = Vertex1DLayout<>::ConstView;

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void init(Vertex1DSoAView &vertices) { vertices.nvFinal() = 0; }

  }  // namespace utilities
}  // namespace vertex1d

#endif  // DataFormats_Vertex1DSoA_interface_alpaka_Vertex1DUtilities_h
