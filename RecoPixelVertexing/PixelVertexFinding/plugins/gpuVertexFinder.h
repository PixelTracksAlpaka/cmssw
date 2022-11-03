#ifndef RecoPixelVertexing_PixelVertexFinding_plugins_gpuVertexFinder_h
#define RecoPixelVertexing_PixelVertexFinding_plugins_gpuVertexFinder_h

#include <cstddef>
#include <cstdint>

//#include "CUDADataFormats/Vertex/interface/ZVertexHeterogeneous.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousHost.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousDevice.h"
#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"

namespace gpuVertexFinder {

  using VtxSoAView = ZVertex::ZVertexSoAView;
  using TkSoAConstView = pixelTrack::TrackSoAConstView;

  // workspace used in the vertex reco algos
  struct WorkSpace {
    static constexpr uint32_t MAXTRACKS = ZVertex::utilities::MAXTRACKS;
    static constexpr uint32_t MAXVTX = ZVertex::utilities::MAXVTX;

    uint32_t ntrks;            // number of "selected tracks"
    uint16_t itrk[MAXTRACKS];  // index of original track
    float zt[MAXTRACKS];       // input track z at bs
    float ezt2[MAXTRACKS];     // input error^2 on the above
    float ptt2[MAXTRACKS];     // input pt^2 on the above
    uint8_t izt[MAXTRACKS];    // interized z-position of input tracks
    int32_t iv[MAXTRACKS];     // vertex index for each associated track

    uint32_t nvIntermediate;  // the number of vertices after splitting pruning etc.

    __host__ __device__ void init() {
      ntrks = 0;
      nvIntermediate = 0;
    }
  };

  __global__ void init(VtxSoAView pdata, WorkSpace* pws) {
    ZVertex::utilities::init(pdata);
    pws->init();
  }

  class Producer {
  public:
    using VtxSoAView = ZVertex::ZVertexSoAView;
    using WorkSpace = gpuVertexFinder::WorkSpace;

    Producer(bool oneKernel,
             bool useDensity,
             bool useDBSCAN,
             bool useIterative,
             int iminT,      // min number of neighbours to be "core"
             float ieps,     // max absolute distance to cluster
             float ierrmax,  // max error to be "seed"
             float ichi2max  // max normalized distance to cluster
             )
        : oneKernel_(oneKernel && !(useDBSCAN || useIterative)),
          useDensity_(useDensity),
          useDBSCAN_(useDBSCAN),
          useIterative_(useIterative),
          minT(iminT),
          eps(ieps),
          errmax(ierrmax),
          chi2max(ichi2max) {}

    ~Producer() = default;

    ZVertex::ZVertexSoADevice makeAsync(cudaStream_t stream, TkSoAConstView tracks_view, float ptMin, float ptMax) const;
    ZVertex::ZVertexSoAHost make(TkSoAConstView tracks_view, float ptMin, float ptMax) const;

  private:
    const bool oneKernel_;
    const bool useDensity_;
    const bool useDBSCAN_;
    const bool useIterative_;

    int minT;       // min number of neighbours to be "core"
    float eps;      // max absolute distance to cluster
    float errmax;   // max error to be "seed"
    float chi2max;  // max normalized distance to cluster
  };

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_plugins_gpuVertexFinder_h
