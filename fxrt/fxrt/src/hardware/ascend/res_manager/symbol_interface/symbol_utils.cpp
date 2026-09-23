#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"
#include <string>
#include "hardware/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_compiler_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_mdl_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_op_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_allocator_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_tdt_symbol.h"

namespace fxrt::device::ascend {

static bool loadAscendApi = false;
static bool loadSimulationApi = false;
static const char* socVersion = nullptr;

void* GetLibHandler(const std::string& libPath, bool ifGlobal) {
  void* handler = nullptr;
  if (ifGlobal) {
    handler = dlopen(libPath.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  } else {
    handler = dlopen(libPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
  }
  if (handler == nullptr) {
    RT_VLOG(VL_HARDWARE) << "Dlopen " << libPath << " failed!" << dlerror();
  }
  return handler;
}

std::string GetAscendPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(aclrtMalloc), &info) == 0) {
    RT_GLOG(ERROR) << "Get dladdr failed.";
    return "";
  }
  auto pathTmp = std::string(info.dli_fname);
  const char kSlash[] = "/";
  auto pos1 = pathTmp.rfind(kSlash);
  if (pos1 != std::string::npos) {
    auto pos2 = pathTmp.rfind(kSlash, pos1 - 1);
    if (pos2 != std::string::npos) {
      return pathTmp.substr(0, pos2) + kSlash;
    }
  }

  RT_GLOG(ERROR) << "Get ascend path based on aclrtMalloc file " << pathTmp
                 << " failed, please check whether CANN packages are installed correctly, \n"
                    "and environment variables are set by source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh.";
  return "";
}

const char* GetAscendSocVersion() {
  if (socVersion != nullptr) {
    return socVersion;
  }
  socVersion = CALL_ASCEND_API(aclrtGetSocName);
  return socVersion;
}

void LoadAscendApiSymbols() {
  if (loadAscendApi) {
    RT_VLOG(VL_HARDWARE) << "Ascend api is already loaded.";
    return;
  }
  std::string ascendPath = GetAscendPath();
  LoadAclBaseApiSymbol(ascendPath);
  LoadAclOpCompilerApiSymbol(ascendPath);
  LoadAclMdlApiSymbol(ascendPath);
  LoadAclOpApiSymbol(ascendPath);
  LoadAclAllocatorApiSymbol(ascendPath);
  LoadAclRtApiSymbol(ascendPath);
  LoadAclApiSymbol(ascendPath);
  LoadAcltdtApiSymbol(ascendPath);
  loadAscendApi = true;
  RT_VLOG(VL_HARDWARE) << "Load ascend api success!";
}

void LoadSimulationApiSymbols() {
  if (loadSimulationApi) {
    RT_VLOG(VL_HARDWARE) << "Simulation api is already loaded.";
    return;
  }

  LoadSimulationAclBaseApi();
  LoadSimulationRtApi();
  LoadSimulationTdtApi();
  LoadSimulationAclOpCompilerApi();
  LoadSimulationAclMdlApi();
  LoadSimulationAclOpApi();
  LoadSimulationAclAllocatorApi();
  LoadSimulationAclApi();
  loadSimulationApi = true;
  RT_VLOG(VL_HARDWARE) << "Load simulation api success!";
}
} // namespace fxrt::device::ascend
