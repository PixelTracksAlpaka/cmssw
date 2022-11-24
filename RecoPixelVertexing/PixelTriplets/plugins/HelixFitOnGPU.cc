#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HelixFitOnGPU.h"

void HelixFitOnGPU::allocateOnGPU(TupleMultiplicity const *tupleMultiplicity, OutputSoAView helix_fit_results) {
  tuples_ = &helix_fit_results.hitIndices();
  tupleMultiplicity_ = tupleMultiplicity;
  outputSoa_ = helix_fit_results;

  assert(tuples_);
  assert(tupleMultiplicity_);
  // assert(outputSoa_); // TODO find equivalent assertion for View
}

void HelixFitOnGPU::deallocateOnGPU() {}
