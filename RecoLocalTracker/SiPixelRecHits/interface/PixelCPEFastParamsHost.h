#ifndef DataFormats_PixelCPEFastParams_interface_PixelCPEFastParamsHost_h
#define DataFormats_PixelCPEFastParams_interface_PixelCPEFastParamsHost_h

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/TrackingRecHitSoA/interface/SiPixelHitStatus.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGenericBase.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelGenError.h"

#include "pixelCPEforDevice.h"

namespace pixelCPEforDevice {

constexpr float micronsToCm = 1.0e-4;

}

/*! \file PixelCPEFastParamsHost.h
 * ------------------------------------------------------
 * Algorithm to estimate cluster positions on the host (CPU)
 * using the generic algorithm. It calls the same function as
 * if running on device so it uses the same SoA formats
 * ------------------------------------------------------
 */


template <typename TrackerTraits>
class PixelCPEFastParamsHost : public PixelCPEGenericBase {
  
   public:
    using Buffer = cms::alpakatools::host_buffer<pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>>;
    using ConstBuffer = cms::alpakatools::const_host_buffer<pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>>;

    /*! \fn PixelCPEFastParamsHost
     * Constructor runs the fillParamsForDevice() to load all
     * the geometry parameters and read the error parameters
     * from the database.
     */
    PixelCPEFastParamsHost(edm::ParameterSet const& conf,
                                          const MagneticField* mag,
                                          const TrackerGeometry& geom,
                                          const TrackerTopology& ttopo,
                                          const SiPixelLorentzAngle* lorentzAngle,
                                          const SiPixelGenErrorDBObject* genErrorDBObject,
                                          const SiPixelLorentzAngle* lorentzAngleWidth);
                                          
    Buffer buffer() { return buffer_; }
    ConstBuffer buffer() const { return buffer_; }
    ConstBuffer const_buffer() const { return buffer_; }
    // pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>* data() const { return buffer_.data(); }
    auto size() const { return alpaka::getExtentProduct(buffer_); }

    static void fillPSetDescription(edm::ParameterSetDescription &desc);

  private:

    /*! \fn localPosition
     * Returns LocalPoint of a given cluster (hit) in the local frame in cm.
     */
    LocalPoint localPosition(DetParam const &theDetParam, ClusterParam &theClusterParam) const override;

    /*! \fn localError
     * Reads and returns cluster position uncertainty in the form of LocalError, (xerr_sq, 0, yerr_sq) in cm^2.
     * The calculation of the error happens during the call to localPosition()
     */
    LocalError localError(DetParam const &theDetParam, ClusterParam &theClusterParam) const override;

    /*! \fn errorFromTemplates
     * Sets the cluster parameters related to errors to the values read out from the templates.
     * The heavy lifting is done by the qbin function.
     */
    void errorFromTemplates(DetParam const &theDetParam, ClusterParamGeneric &theClusterParam, float qclus) const;

    std::vector<SiPixelGenErrorStore> thePixelGenError_;
    
    /*! \fn fillParamsForDevice
     * Fills the buffer with SoA "ParamsOnDeviceT" object constructed from corresponding values from DetParams AoS. 
     * The SoA is used when calculating positions and errors on device.
     */
    void fillParamsForDevice();

    Buffer buffer_;

};
// }  // namespace pixelCPEforDevice

#endif
