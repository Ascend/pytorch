#include "acl_rt_allocator_symbol.h"
#include <string>
#include "symbol_utils.h"

namespace fxrt::device::ascend {
aclrtAllocatorCreateDescFunObj aclrtAllocatorCreateDesc_ = nullptr;
aclrtAllocatorDestroyDescFunObj aclrtAllocatorDestroyDesc_ = nullptr;
aclrtAllocatorRegisterFunObj aclrtAllocatorRegister_ = nullptr;
aclrtAllocatorSetAllocAdviseFuncToDescFunObj aclrtAllocatorSetAllocAdviseFuncToDesc_ = nullptr;
aclrtAllocatorSetAllocFuncToDescFunObj aclrtAllocatorSetAllocFuncToDesc_ = nullptr;
aclrtAllocatorSetFreeFuncToDescFunObj aclrtAllocatorSetFreeFuncToDesc_ = nullptr;
aclrtAllocatorSetGetAddrFromBlockFuncToDescFunObj aclrtAllocatorSetGetAddrFromBlockFuncToDesc_ = nullptr;
aclrtAllocatorSetObjToDescFunObj aclrtAllocatorSetObjToDesc_ = nullptr;
aclrtAllocatorUnregisterFunObj aclrtAllocatorUnregister_ = nullptr;

void LoadAclAllocatorApiSymbol(const std::string& ascendPath) {
  std::string allocatorPluginPath = ascendPath + "lib64/libascendcl.so";
  auto handler = GetLibHandler(allocatorPluginPath);
  if (handler == nullptr) {
    RT_VLOG(VL_HARDWARE) << "Dlopen " << allocatorPluginPath << " failed!" << dlerror();
    return;
  }
  aclrtAllocatorCreateDesc_ = DlsymAscendFuncObj(aclrtAllocatorCreateDesc, handler);
  aclrtAllocatorDestroyDesc_ = DlsymAscendFuncObj(aclrtAllocatorDestroyDesc, handler);
  aclrtAllocatorRegister_ = DlsymAscendFuncObj(aclrtAllocatorRegister, handler);
  aclrtAllocatorSetAllocAdviseFuncToDesc_ = DlsymAscendFuncObj(aclrtAllocatorSetAllocAdviseFuncToDesc, handler);
  aclrtAllocatorSetAllocFuncToDesc_ = DlsymAscendFuncObj(aclrtAllocatorSetAllocFuncToDesc, handler);
  aclrtAllocatorSetFreeFuncToDesc_ = DlsymAscendFuncObj(aclrtAllocatorSetFreeFuncToDesc, handler);
  aclrtAllocatorSetGetAddrFromBlockFuncToDesc_ =
      DlsymAscendFuncObj(aclrtAllocatorSetGetAddrFromBlockFuncToDesc, handler);
  aclrtAllocatorSetObjToDesc_ = DlsymAscendFuncObj(aclrtAllocatorSetObjToDesc, handler);
  aclrtAllocatorUnregister_ = DlsymAscendFuncObj(aclrtAllocatorUnregister, handler);
  RT_VLOG(VL_HARDWARE) << "Load acl allocator api success!";
}

void LoadSimulationAclAllocatorApi() {
  ASSIGN_SIMU(aclrtAllocatorCreateDesc);
  ASSIGN_SIMU(aclrtAllocatorDestroyDesc);
  ASSIGN_SIMU(aclrtAllocatorRegister);
  ASSIGN_SIMU(aclrtAllocatorSetAllocAdviseFuncToDesc);
  ASSIGN_SIMU(aclrtAllocatorSetAllocFuncToDesc);
  ASSIGN_SIMU(aclrtAllocatorSetFreeFuncToDesc);
  ASSIGN_SIMU(aclrtAllocatorSetGetAddrFromBlockFuncToDesc);
  ASSIGN_SIMU(aclrtAllocatorSetObjToDesc);
  ASSIGN_SIMU(aclrtAllocatorUnregister);
}
} // namespace fxrt::device::ascend
