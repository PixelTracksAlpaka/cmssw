#ifndef CalibTracker_Records_interface_SiPixelRecords_h
#define CalibTracker_Records_interface_SiPixelRecords_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class PixelESTestRecord
    : public edm::eventsetup::DependentRecordImplementation<PixelESTestRecord,
                                                            edm::mpl::Vector<TrackerDigiGeometryRecord>> {
};

#endif
