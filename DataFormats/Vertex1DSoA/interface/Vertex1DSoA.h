#ifndef DataFormats_Vertex1DSoA_interface_Vertex1DSoA_h
#define DataFormats_Vertex1DSoA_interface_Vertex1DSoA_h

#include <Eigen/Core>
#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(Vertex1DLayout,
                    SOA_COLUMN(int16_t, idv),
                    SOA_COLUMN(float, zv),
                    SOA_COLUMN(float, wv),
                    SOA_COLUMN(float, chi2),
                    SOA_COLUMN(float, ptv2),
                    SOA_COLUMN(int32_t, ndof),
                    SOA_COLUMN(uint16_t, sortInd),
                    SOA_SCALAR(uint32_t, nvFinal))

// Previous Vertex1DSoA class methods.
// They operate on View and ConstView of the Vertex1DSoA.
namespace vertex1d {
  // Common types for both Host and Device code
  using Vertex1DSoA = Vertex1DLayout<>;
  using Vertex1DSoAView = Vertex1DLayout<>::View;
  using Vertex1DSoAConstView = Vertex1DLayout<>::ConstView;

}  // namespace vertex1d

#endif  // DataFormats_Vertex1DSoA_interface_Vertex1DSoA_h
