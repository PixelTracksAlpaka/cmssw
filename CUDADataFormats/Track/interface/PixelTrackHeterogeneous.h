#ifndef CUDADataFormats_Track_PixelTrackHeterogeneous_h
#define CUDADataFormats_Track_PixelTrackHeterogeneous_h

#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
// #include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT_test.h"

using PixelTrackHeterogeneous = HeterogeneousSoA<pixelTrack::TrackSoA>;

#endif  // #ifndef CUDADataFormats_Track_PixelTrackHeterogeneous_h
