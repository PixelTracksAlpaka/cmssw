/*#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT.h"

#include <iostream>
#include <cassert>

int main() {
  // test quality

  auto q = pixelTrackSoA::qualityByName("tight");
  assert(pixelTrackSoA::Quality::tight == q);
  q = pixelTrackSoA::qualityByName("toght");
  assert(pixelTrackSoA::Quality::notQuality == q);

  for (uint32_t i = 0; i < pixelTrackSoA::qualitySize; ++i) {
    auto const qt = static_cast<pixelTrackSoA::Quality>(i);
    auto q = pixelTrackSoA::qualityByName(pixelTrackSoA::qualityName[i]);
    assert(qt == q);
  }

  return 0;
}*/
