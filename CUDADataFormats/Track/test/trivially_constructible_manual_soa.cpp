#include <bits/stdc++.h>
#include <type_traits>
#include <stdint.h>
#include <assert.h>
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT.h"

using namespace std;

int main() {
  std::cout << "pixelTrack::TrackSoA with manually defined SoA, trivially constructible: "
            << std::is_trivially_constructible<pixelTrack::TrackSoA>::value << std::endl;

  return 0;
}
