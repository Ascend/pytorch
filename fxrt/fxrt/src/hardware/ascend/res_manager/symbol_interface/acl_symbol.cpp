#include "acl_symbol.h"
#include <string>
#include "symbol_utils.h"

namespace fxrt::device::ascend {

aclInitFunObj aclInit_ = nullptr;
aclFinalizeFunObj aclFinalize_ = nullptr;

void LoadAclApiSymbol(const std::string& ascendPath) {
  std::string aclPluginPath = ascendPath + "lib64/libascendcl.so";
  auto baseHandler = GetLibHandler(aclPluginPath);
  if (baseHandler == nullptr) {
    RT_VLOG(VL_HARDWARE) << "Dlopen " << aclPluginPath << " failed!" << dlerror();
    return;
  }
  aclInit_ = DlsymAscendFuncObj(aclInit, baseHandler);
  aclFinalize_ = DlsymAscendFuncObj(aclFinalize, baseHandler);
  RT_VLOG(VL_HARDWARE) << "Load acl base api success!";
}

void LoadSimulationAclApi() {
  ASSIGN_SIMU(aclInit);
  ASSIGN_SIMU(aclFinalize);
}
} // namespace fxrt::device::ascend
