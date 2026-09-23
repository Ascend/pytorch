#ifndef FXRT_SRC_HARDWARE_ASCEND_ACL_RT_ALLOCATOR_SYMBOL_H_
#define FXRT_SRC_HARDWARE_ASCEND_ACL_RT_ALLOCATOR_SYMBOL_H_
#include <string>
#include "acl/acl_rt_allocator.h"
#include "hardware/hardware_abstract/dlopen_macro.h"

namespace fxrt::device::ascend {
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorCreateDesc, aclrtAllocatorDesc)
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorDestroyDesc, aclError, aclrtAllocatorDesc)
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorRegister, aclError, aclrtStream, aclrtAllocatorDesc)
ORIGIN_METHOD_WITH_SIMU(
    aclrtAllocatorSetAllocAdviseFuncToDesc,
    aclError,
    aclrtAllocatorDesc,
    aclrtAllocatorAllocAdviseFunc)
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorSetAllocFuncToDesc, aclError, aclrtAllocatorDesc, aclrtAllocatorAllocFunc)
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorSetFreeFuncToDesc, aclError, aclrtAllocatorDesc, aclrtAllocatorFreeFunc)
ORIGIN_METHOD_WITH_SIMU(
    aclrtAllocatorSetGetAddrFromBlockFuncToDesc,
    aclError,
    aclrtAllocatorDesc,
    aclrtAllocatorGetAddrFromBlockFunc)
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorSetObjToDesc, aclError, aclrtAllocatorDesc, aclrtAllocator)
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorUnregister, aclError, aclrtStream)

extern aclrtAllocatorCreateDescFunObj aclrtAllocatorCreateDesc_;
extern aclrtAllocatorDestroyDescFunObj aclrtAllocatorDestroyDesc_;
extern aclrtAllocatorRegisterFunObj aclrtAllocatorRegister_;
extern aclrtAllocatorSetAllocAdviseFuncToDescFunObj aclrtAllocatorSetAllocAdviseFuncToDesc_;
extern aclrtAllocatorSetAllocFuncToDescFunObj aclrtAllocatorSetAllocFuncToDesc_;
extern aclrtAllocatorSetFreeFuncToDescFunObj aclrtAllocatorSetFreeFuncToDesc_;
extern aclrtAllocatorSetGetAddrFromBlockFuncToDescFunObj aclrtAllocatorSetGetAddrFromBlockFuncToDesc_;
extern aclrtAllocatorSetObjToDescFunObj aclrtAllocatorSetObjToDesc_;
extern aclrtAllocatorUnregisterFunObj aclrtAllocatorUnregister_;

void LoadAclAllocatorApiSymbol(const std::string& ascendPath);
void LoadSimulationAclAllocatorApi();
} // namespace fxrt::device::ascend

#endif // FXRT_SRC_HARDWARE_ASCEND_ACL_RT_ALLOCATOR_SYMBOL_H_
