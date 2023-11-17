#ifndef DataFormats_PortableTestObjects_interface_alpaka_TestDeviceObject_h
#define DataFormats_PortableTestObjects_interface_alpaka_TestDeviceObject_h

#include "DataFormats/Portable/interface/alpaka/PortableObject.h"
#include "DataFormats/PortableTestObjects/interface/TestStruct.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace portabletest {

    // make the names from the top-level portabletest namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::portabletest namespace
    using namespace ::portabletest;

    // struct with x, y, z, id fields in device global memory
    using TestDeviceObject = PortableObject<TestStruct>;

  }  // namespace portabletest

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_PortableTestObjects_interface_alpaka_TestDeviceObject_h
