#ifndef DataFormats_PixelTrackSoA_interface_PixelTrackDefinitions_h
#define DataFormats_PixelTrackSoA_interface_PixelTrackDefinitions_h
#include <string>
#include <algorithm>
#include <stdexcept>

namespace pixelTrackSoA { // TODO: once the CUDA code is removed, this should be changed back to pixelTrack

  enum class Quality : uint8_t { bad = 0, edup, dup, loose, strict, tight, highPurity, notQuality };
  constexpr uint32_t qualitySize{uint8_t(Quality::notQuality)};
  constexpr std::string_view qualityName[qualitySize]{"bad", "edup", "dup", "loose", "strict", "tight", "highPurity"};
  inline Quality qualityByName(std::string const &name) {
    auto qp = std::find(qualityName, qualityName + qualitySize, name) - qualityName;
    auto ret = static_cast<Quality>(qp);
    if (ret == pixelTrackSoA::Quality::notQuality)
      throw std::invalid_argument(name + "is not a pixelTrackSoA::Quality!");

    return ret;
  }

#ifdef GPU_SMALL_EVENTS
  // kept for testing and debugging
  constexpr uint32_t maxNumber() { return 2 * 1024; }
#else
  // tested on MC events with 55-75 pileup events
  constexpr uint32_t maxNumber() { return 32 * 1024; }
#endif

}  // namespace pixelTrackSoA

#endif  // DataFormats_PixelTrackSoA_interface_PixelTrackDefinitions_h
