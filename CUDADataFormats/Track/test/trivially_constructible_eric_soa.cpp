#include <bits/stdc++.h>
#include <type_traits>
#include <stdint.h>
#include <assert.h>
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT_test.h"

using namespace std;

int main() {
  std::cout << "pixelTrack::TrackSoA with Eric's SoA, trivially constructible: "
            << std::is_trivially_constructible<pixelTrack::TrackSoA>::value << std::endl;

  std::cout << "cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>> trivially constructible: "
            << std::is_trivially_constructible<cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousT_test<>>>::value
            << std::endl;

  return 0;
}
