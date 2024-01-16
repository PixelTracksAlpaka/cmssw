#ifndef DataFormats_TrackerRecHitSoA_interface_SiPixelRecHitStatus_h
#define DataFormats_TrackerRecHitSoA_interface_SiPixelRecHitStatus_h

#include <cstdint>

// more information on bit fields : https://en.cppreference.com/w/cpp/language/bit_field
struct SiPixelRecHitStatus {
  bool isBigX : 1;   //  ∈[0,1]
  bool isOneX : 1;   //  ∈[0,1]
  bool isBigY : 1;   //  ∈[0,1]
  bool isOneY : 1;   //  ∈[0,1]
  uint8_t qBin : 3;  //  ∈[0,1,...,7]
};

struct SiPixelRecHitStatusAndCharge {
  SiPixelRecHitStatus status;
  uint32_t charge : 24;
};

#endif  // DataFormats_TrackerRecHitSoA_interface_SiPixelRecHitStatus_h
