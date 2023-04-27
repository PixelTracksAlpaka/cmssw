#ifndef CalibTracker_SiPixelESProducers_SiPixelMappingSoARecord_h
#define CalibTracker_SiPixelESProducers_SiPixelMappingSoARecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

// class SiPixelMappingSoARecord : public edm::eventsetup::EventSetupRecordImplementation<SiPixelMappingSoARecord> {};

class SiPixelMappingSoARecord
    : public edm::eventsetup::DependentRecordImplementation<
          SiPixelMappingSoARecord,
          edm::mpl::Vector<SiPixelFedCablingMapRcd, SiPixelQualityRcd, TrackerDigiGeometryRecord>> {};

#endif