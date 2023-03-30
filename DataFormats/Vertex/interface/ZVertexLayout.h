#ifndef DataFormats_Vertex_ZVertexLayout_h
#define DataFormats_Vertex_ZVertexLayout_h

<<<<<<< HEAD
=======
#include <Eigen/Core>
>>>>>>> common/alpaka_port_13_1_dev
#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(ZVertexSoAHeterogeneousLayout,
                    SOA_COLUMN(int16_t, idv),
                    SOA_COLUMN(float, zv),
                    SOA_COLUMN(float, wv),
                    SOA_COLUMN(float, chi2),
                    SOA_COLUMN(float, ptv2),
                    SOA_COLUMN(int32_t, ndof),
                    SOA_COLUMN(uint16_t, sortInd),
                    SOA_SCALAR(uint32_t, nvFinal))

// Previous ZVertexSoA class methods.
// They operate on View and ConstView of the ZVertexSoA.
namespace zVertex {
  // Common types for both Host and Device code
  using ZVertexSoALayout = ZVertexSoAHeterogeneousLayout<>;
  using ZVertexSoAView = ZVertexSoAHeterogeneousLayout<>::View;
  using ZVertexSoAConstView = ZVertexSoAHeterogeneousLayout<>::ConstView;

}  // namespace zVertex

#endif
