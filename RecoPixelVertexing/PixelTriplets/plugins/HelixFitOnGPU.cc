#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HelixFitOnGPU.h"

void HelixFitOnGPU::allocateOnGPU(Tuples const *tuples,
                                  TupleMultiplicity const *tupleMultiplicity,
                                  OutputSoAView helix_fit_results) {
  tuples_ = tuples;
  tupleMultiplicity_ = tupleMultiplicity;
  outputSoa_ = helix_fit_results;

  assert(tuples_);
  assert(tupleMultiplicity_);
  // assert(outputSoa_); // TODO find equivalent assertion for View
}

void HelixFitOnGPU::deallocateOnGPU() {}
