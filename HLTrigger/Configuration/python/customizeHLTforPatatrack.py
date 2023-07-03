import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.common import *

def customiseHLTforTestingDQMGPUvsCPU(process):
    '''Ad-hoc changes to test HLT config containing only DQM_PixelReconstruction_v and DQMGPUvsCPU stream
    '''
    if hasattr(process, 'hltDatasetDQMGPUvsCPU'):
        process.hltDatasetDQMGPUvsCPU.triggerConditions = ['DQM_PixelReconstruction_v*']

    # remove FinalPaths running OutputModules, except for the DQMGPUvsCPU one
    finalPathsToRemove = []
    for fpath in process.finalpaths_():
        if 'DQMGPUvsCPU' not in fpath:
            finalPathsToRemove += [fpath]
    for fpath in finalPathsToRemove:
        process.__delattr__(fpath)

#    # add DQMIO output file
#    if not hasattr(process, 'DQMStore'):
#        process.load("DQMServices.Core.DQMStore_cfi")
#
#    if not hasattr(process, 'dqmOutput'):
#        process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
#            fileName = cms.untracked.string("DQMIO.root")
#        )
#
#    if not hasattr(process, 'DQMOutput'):
#        process.DQMOutput = cms.FinalPath( process.dqmOutput )
#        process.schedule.append( process.DQMOutput )

    return process

def customiseHLTforTestingDQMGPUvsCPUPixelOnlyUpToLocal(process):
    '''Ad-hoc changes to test HLT config containing only DQM_PixelReconstruction_v and DQMGPUvsCPU stream
       only up to the Pixel Local Reconstruction
    '''
    process = customiseHLTforTestingDQMGPUvsCPU(process)

    if not hasattr(process, 'HLTDoLocalPixelTask'):
        return process

    process.hltPixelConsumerGPU.eventProducts = [
        'hltSiPixelClusters',
        'hltSiPixelClustersLegacyFormat',
        'hltSiPixelDigiErrorsLegacyFormat',
        'hltSiPixelRecHits',
        'hltSiPixelRecHitsLegacyFormat',
        'hltPixelTracks',
    ]

    process.hltPixelConsumerCPU.eventProducts = []
#    for foo in process.hltPixelConsumerGPU.eventProducts:
#        process.hltPixelConsumerCPU.eventProducts += [foo+'CPUSerial']

    # modify EventContent of DQMGPUvsCPU stream
    if hasattr(process, 'hltOutputDQMGPUvsCPU'):
        process.hltOutputDQMGPUvsCPU.outputCommands = [
            'drop *',
            'keep *Cluster*_hltSiPixelClusters_*_*',
            'keep *Cluster*_hltSiPixelClustersCPUSerial_*_*',
            'keep *_hltSiPixelDigiErrorsLegacyFormat_*_*',
            'keep *_hltSiPixelDigiErrorsLegacyFormatCPUSerial_*_*',
#            'keep *RecHit*_hltSiPixelRecHits_*_*',
#            'keep *RecHit*_hltSiPixelRecHitsCPUSerial_*_*',
        ]

    # empty HLTRecopixelvertexingSequence until we add tracks and vertices
    process.HLTRecopixelvertexingSequence = cms.Sequence()

    # create CPU version of LocalPixelRecoSequence, and add it to HLTDQMPixelReconstruction
    process.HLTDoLocalPixelSequenceCPUSerial = cms.Sequence( process.HLTDoLocalPixelTaskCPUSerial )
    process.HLTDQMPixelReconstruction.insert(0, process.HLTDoLocalPixelSequenceCPUSerial)

    return process

def customiseHLTforAlpakaPixelRecoLocal(process):
    '''Customisation to introduce the Local Pixel Reconstruction in Alpaka
    '''
    process.hltESPSiPixelCablingSoA = cms.ESProducer('SiPixelCablingSoAESProducer@alpaka',
        ComponentName = cms.string(''),
        CablingMapLabel = cms.string(''),
        UseQualityInfo = cms.bool(False),
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltESPSiPixelGainCalibrationForHLTSoA = cms.ESProducer('SiPixelGainCalibrationForHLTSoAESProducer@alpaka',
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltESPPixelCPEFastParamsPhase1 = cms.ESProducer('PixelCPEFastParamsESProducerAlpakaPhase1@alpaka',
        ComponentName = cms.string('PixelCPEFast'),
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    ###

    # alpaka EDProducer
    # consumes
    #  - reco::BeamSpot
    # produces
    #  - BeamSpotDeviceProduct
    process.hltOnlineBeamSpotDevice = cms.EDProducer("BeamSpotDeviceProducer@alpaka",
        src = cms.InputTag("hltOnlineBeamSpot"),
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    # alpaka EDProducer
    # consumes
    #  - FEDRawDataCollection
    # produces (* optional)
    #  - SiPixelClustersSoA
    #  - SiPixelDigisCollection
    #  - SiPixelDigiErrorsCollection *
    #  - SiPixelFormatterErrors *
    process.hltSiPixelClusters = cms.EDProducer('SiPixelRawToCluster@alpaka',
        isRun2 = cms.bool(False),
        IncludeErrors = cms.bool(True),
        UseQualityInfo = cms.bool(False),
        clusterThreshold_layer1 = cms.int32(4000),
        clusterThreshold_otherLayers = cms.int32(4000),
        InputLabel = cms.InputTag('rawDataCollector'),
        Regions = cms.PSet(
            inputs = cms.optional.VInputTag,
            deltaPhi = cms.optional.vdouble,
            maxZ = cms.optional.vdouble,
            beamSpot = cms.optional.InputTag
        ),
        CablingMapLabel = cms.string(''),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltSiPixelClustersLegacyFormat = cms.EDProducer('SiPixelDigisClustersFromSoAAlpakaPhase1',
        src = cms.InputTag('hltSiPixelClusters'),
        clusterThreshold_layer1 = cms.int32(4000),
        clusterThreshold_otherLayers = cms.int32(4000),
        produceDigis = cms.bool(False),
        storeDigis = cms.bool(False)
    )

    process.hltSiPixelClustersCache = cms.EDProducer("SiPixelClusterShapeCacheProducer",
        src = cms.InputTag( "hltSiPixelClustersLegacyFormat" ),
        onDemand = cms.bool( False )
    )

    # legacy EDProducer
    # consumes
    #  - SiPixelDigiErrorsHost
    #  - SiPixelFormatterErrors
    # produces
    #  - edm::DetSetVector<SiPixelRawDataError>
    #  - DetIdCollection
    #  - DetIdCollection, "UserErrorModules"
    #  - edmNew::DetSetVector<PixelFEDChannel>
    process.hltSiPixelDigiErrorsLegacyFormat = cms.EDProducer("SiPixelDigiErrorsFromSoA",
        digiErrorSoASrc = cms.InputTag("hltSiPixelClusters"),
        fmtErrorsSoASrc = cms.InputTag("hltSiPixelClusters"),
        CablingMapLabel = cms.string(''),
        UsePhase1 = cms.bool(True),
        ErrorList = cms.vint32(29),
        UserErrorList = cms.vint32(40)
    )

    # alpaka EDProducer
    # consumes
    #  - BeamSpotDeviceProduct
    #  - SiPixelClustersSoA
    #  - SiPixelDigisCollection
    # produces
    #  - TrackingRecHitAlpakaCollection<TrackerTraits>
    process.hltSiPixelRecHits = cms.EDProducer("SiPixelRecHitAlpakaPhase1@alpaka",
        beamSpot = cms.InputTag('hltOnlineBeamSpotDevice'),
        src = cms.InputTag('hltSiPixelClusters'),
        CPE = cms.string('PixelCPEFast'),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltSiPixelRecHitsLegacyFormat = cms.EDProducer('SiPixelRecHitFromSoAAlpakaPhase1',
        pixelRecHitSrc = cms.InputTag('hltSiPixelRecHits'),
        src = cms.InputTag('hltSiPixelClustersLegacyFormat'),
    )

    ###
    ### Task: Pixel Local Reconstruction
    ###
    process.HLTDoLocalPixelTask = cms.ConditionalTask(
        process.hltOnlineBeamSpotDevice,
        process.hltSiPixelClusters,
        process.hltSiPixelClustersLegacyFormat,   # was: hltSiPixelClusters
        process.hltSiPixelClustersCache,          # really needed ??
        process.hltSiPixelDigiErrorsLegacyFormat, # was: hltSiPixelDigis
        process.hltSiPixelRecHits,
        process.hltSiPixelRecHitsLegacyFormat,    # was: hltSiPixelRecHits
        process.hltPixelTracks,
    )

    ###
    ### CPUSerial version of Pixel Local Reconstruction
    ###
    process.hltOnlineBeamSpotDeviceCPUSerial = process.hltOnlineBeamSpotDevice.clone(
        alpaka = dict( backend = 'serial_sync' )
    )

    process.hltSiPixelClustersCPUSerial = process.hltSiPixelClusters.clone(
        alpaka = dict( backend = 'serial_sync' )
    )

    process.hltSiPixelClustersLegacyFormatCPUSerial = process.hltSiPixelClustersLegacyFormat.clone(
        src = 'hltSiPixelClustersCPUSerial'
    )

    process.hltSiPixelDigiErrorsLegacyFormatCPUSerial = process.hltSiPixelDigiErrorsLegacyFormat.clone(
        digiErrorSoASrc = "hltSiPixelClustersCPUSerial",
        fmtErrorsSoASrc = "hltSiPixelClustersCPUSerial",
    )

    process.hltSiPixelRecHitsCPUSerial = process.hltSiPixelRecHits.clone(
        beamSpot = 'hltOnlineBeamSpotDeviceCPUSerial',
        src = 'hltSiPixelClustersCPUSerial',
        alpaka = dict( backend = 'serial_sync' )
    )

    process.hltSiPixelRecHitsLegacyFormatCPUSerial = process.hltSiPixelRecHitsLegacyFormat.clone(
        pixelRecHitSrc = 'hltSiPixelRecHitsCPUSerial',
        src = 'hltSiPixelClustersLegacyFormatCPUSerial',
    )

    process.HLTDoLocalPixelTaskCPUSerial = cms.ConditionalTask(
        process.hltOnlineBeamSpotDeviceCPUSerial,
        process.hltSiPixelClustersCPUSerial,
        process.hltSiPixelClustersLegacyFormatCPUSerial,
        process.hltSiPixelDigiErrorsLegacyFormatCPUSerial,
        process.hltSiPixelRecHitsCPUSerial,
        process.hltSiPixelRecHitsLegacyFormatCPUSerial,
    )

    return process

def customiseHLTforAlpakaPixelRecoTracking(process):
    '''Customisation to introduce the Pixel-Track Reconstruction in Alpaka
    '''

    # alpaka EDProducer
    # consumes
    #  - TrackingRecHitAlpakaCollection<TrackerTraits>
    # produces
    #  - TkSoADevice
    process.hltPixelTracks = cms.EDProducer("CAHitNtupletAlpakaPhase1@alpaka",
        pixelRecHitSrc = cms.InputTag('hltSiPixelRecHits'),
        ptmin = cms.double(0.89999997615814209),
        CAThetaCutBarrel = cms.double(0.0020000000949949026),
        CAThetaCutForward = cms.double(0.0030000000260770321),
        hardCurvCut = cms.double(0.032840722495894911),
        dcaCutInnerTriplet = cms.double(0.15000000596046448),
        dcaCutOuterTriplet = cms.double(0.25),
        earlyFishbone = cms.bool(True),
        lateFishbone = cms.bool(False),
        fillStatistics = cms.bool(False),
        minHitsPerNtuplet = cms.uint32(3),
        maxNumberOfDoublets = cms.uint32(524288),
        minHitsForSharingCut = cms.uint32(10),
        fitNas4 = cms.bool(False),
        doClusterCut = cms.bool(True),
        doZ0Cut = cms.bool(True),
        doPtCut = cms.bool(True),
        useRiemannFit = cms.bool(False),
        doSharedHitCut = cms.bool(True),
        dupPassThrough = cms.bool(False),
        useSimpleTripletCleaner = cms.bool(True),
        idealConditions = cms.bool(False),
        includeJumpingForwardDoublets = cms.bool(True),
        trackQualityCuts = cms.PSet(
            chi2MaxPt = cms.double(10),
            chi2Coeff = cms.vdouble(0.9, 1.8),
            chi2Scale = cms.double(8),
            tripletMinPt = cms.double(0.5),
            tripletMaxTip = cms.double(0.3),
            tripletMaxZip = cms.double(12),
            quadrupletMinPt = cms.double(0.3),
            quadrupletMaxTip = cms.double(0.5),
            quadrupletMaxZip = cms.double(12)
        ),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    return process

def customiseHLTforAlpakaPixelRecoVertexing(process):
    '''Customisation to introduce the Pixel-Vertex Reconstruction in Alpaka
    '''
    # alpaka EDProducer
    # consumes
    #  - TkSoADevice
    # produces
    #  - ZVertexDevice
    process.hltPixelVertices = cms.EDProducer('PixelVertexProducerAlpakaPhase1@alpaka',
        onGPU = cms.bool(True),
        oneKernel = cms.bool(True),
        useDensity = cms.bool(True),
        useDBSCAN = cms.bool(False),
        useIterative = cms.bool(False),
        minT = cms.int32(2),
        eps = cms.double(0.07),
        errmax = cms.double(0.01),
        chi2max = cms.double(9),
        PtMin = cms.double(0.5),
        PtMax = cms.double(75),
        pixelTrackSrc = cms.InputTag('pixelTracksCUDA'),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )
    return process

def customiseHLTforAlpakaPixelReco(process):
    '''Customisation to introduce the Pixel Local+Track+Vertex Reconstruction in Alpaka
    '''
    process.load('Configuration.StandardSequences.Accelerators_cff')
    process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

    process = customiseHLTforAlpakaPixelRecoLocal(process)
    process = customiseHLTforAlpakaPixelRecoTracking(process)
#    process = customiseHLTforAlpakaPixelRecoVertexing(process)
    return process

def customizeHLTforPatatrack(process):
    '''Customise HLT configuration introducing latest Patatrack developments
    '''
    process = customiseHLTforAlpakaPixelReco(process)
    return process
