from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoAAlpakaPhase1_cfi import siPixelDigisClustersFromSoAAlpakaPhase1 as _siPixelDigisClustersFromSoAAlpakaPhase1

import FWCore.ParameterSet.Config as cms

siPixelClustersPreSplitting = cms.EDAlias(
        siPixelDigisClustersPreSplitting = cms.VPSet(
            cms.PSet(type = cms.string("SiPixelClusteredmNewDetSetVector"))
        ))
