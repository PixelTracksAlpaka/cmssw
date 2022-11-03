#ifndef CUDADataFormats_Vertex_ZVertexUtilities_h
#define CUDADataFormats_Vertex_ZVertexUtilities_h

#include <cuda_runtime.h>
#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(ZVertexSoAHeterogeneousLayout,
                    SOA_COLUMN(int16_t, idv),
                    SOA_COLUMN(float, zv),  // this is chi2/ndof as not necessarely all hits are used in the fit
                    SOA_COLUMN(float, wv),
                    SOA_COLUMN(float, chi2),
                    SOA_COLUMN(float, ptv2),
                    SOA_COLUMN(int32_t, ndof),
                    SOA_COLUMN(uint16_t, sortInd),
                    SOA_SCALAR(uint32_t, nvFinal))

// Previous ZVertexSoA class methods.
// They operate on View and ConstView of the ZVertexSoA.
namespace ZVertex {
  namespace utilities {
    using ZVertexSoAView = ZVertexSoAHeterogeneousLayout<>::View;

    static constexpr uint32_t MAXTRACKS = 32 * 1024;
    static constexpr uint32_t MAXVTX = 1024;

    __host__ __device__ inline void init(ZVertexSoAView &vertices) { vertices.nvFinal() = 0; }

  }  // namespace utilities
}  // namespace ZVertex

namespace ZVertex {
  // Common types for both Host and Device code
  using ZVertexSoALayout = ZVertexSoAHeterogeneousLayout<>;
  using ZVertexSoAView = ZVertexSoAHeterogeneousLayout<>::View;
  using ZVertexSoAConstView = ZVertexSoAHeterogeneousLayout<>::ConstView;

}  // namespace ZVertex

#endif
