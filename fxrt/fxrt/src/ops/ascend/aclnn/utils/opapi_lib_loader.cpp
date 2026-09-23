#include "ops/ascend/aclnn/utils/opapi_lib_loader.h"
#include <unistd.h>
#include <string>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace fxrt {
namespace ops {
namespace {
void LoadCommonMetaFuncApi() {
  LOAD_COMMON_META_FUNC(aclCreateTensor);
  LOAD_COMMON_META_FUNC(aclCreateScalar);
  LOAD_COMMON_META_FUNC(aclCreateIntArray);
  LOAD_COMMON_META_FUNC(aclCreateFloatArray);
  LOAD_COMMON_META_FUNC(aclCreateBoolArray);
  LOAD_COMMON_META_FUNC(aclCreateTensorList);

  LOAD_COMMON_META_FUNC(aclDestroyTensor);
  LOAD_COMMON_META_FUNC(aclDestroyScalar);
  LOAD_COMMON_META_FUNC(aclDestroyIntArray);
  LOAD_COMMON_META_FUNC(aclDestroyFloatArray);
  LOAD_COMMON_META_FUNC(aclDestroyBoolArray);
  LOAD_COMMON_META_FUNC(aclDestroyTensorList);
  LOAD_COMMON_META_FUNC(aclDestroyAclOpExecutor);

  LOAD_COMMON_META_FUNC(aclnnInit);
  LOAD_COMMON_META_FUNC(aclnnFinalize);

  LOAD_COMMON_META_FUNC(aclSetAclOpExecutorRepeatable);

  LOAD_COMMON_META_FUNC(aclSetTensorAddr);
  LOAD_COMMON_META_FUNC(aclSetDynamicTensorAddr);
}
} // namespace

static bool isLoaded = false;
static bool isAclnnInit = false;
static std::mutex initMutex;
static std::shared_mutex rwOpApiMutex;
// handler -> libPath
std::unordered_map<void*, std::string> libHandlers;

void LoadOpApiLib() {
  if (isLoaded) {
    return;
  }
  auto customPaths = GetEnv("ASCEND_CUSTOM_OPP_PATH");
  std::vector<std::string> customPathVec;
  if (!customPaths.empty()) {
    RT_VLOG(VL_OPS) << "ASCEND_CUSTOM_OPP_PATH: " << customPaths;

    std::stringstream ss(customPaths);
    std::string path;
    while (std::getline(ss, path, ':')) {
      if (path.empty())
        continue;

      const std::string libPath = path + kNameCustOpApiLib;
      if (access(libPath.c_str(), F_OK) == 0) {
        customPathVec.push_back(libPath);
      }
    }
  }

  const std::string ascendPath = device::ascend::GetAscendPath();
  const std::vector<std::string> dependLibs = {"libdummy_tls.so", "libnnopbase.so"};
  std::unique_lock<std::shared_mutex> writeLock(rwOpApiMutex);
  for (const auto& customLibPath : customPathVec) {
    auto customHandler = GetOpApiLibHandler(customLibPath);
    if (customHandler != nullptr) {
      RT_VLOG(VL_OPS) << "Load cust open api lib " << customLibPath << " success";
      (void)libHandlers.emplace(customHandler, customLibPath);
    }
  }

  for (const auto& depLib : dependLibs) {
    (void)GetOpApiLibHandler(ascendPath + "lib64/" + depLib);
  }
  auto opApiLibPath = ascendPath + kNameOpApiLib;
  auto handler = GetOpApiLibHandler(opApiLibPath);
  if (handler != nullptr) {
    RT_VLOG(VL_OPS) << "Load lib " << opApiLibPath << " success";
    (void)libHandlers.emplace(handler, opApiLibPath);
  }
  LoadCommonMetaFuncApi();
  isLoaded = true;
  RT_VLOG(VL_OPS) << "Load opapi lib success";
}

void* GetAclnnOpApiFunc(const char* apiName) {
  // apiName -> api
  static thread_local std::unordered_map<std::string, void*> opapiCache;
  auto iter = opapiCache.find(std::string(apiName));
  if (iter != opapiCache.end()) {
    RT_VLOG(VL_OPS) << "OpApi " << apiName << " hit cache";
    return iter->second;
  }
  std::shared_lock<std::shared_mutex> readLock(rwOpApiMutex);
  if (libHandlers.size() == 0) {
    readLock.unlock();
    LoadOpApiLib();
  }
  for (auto& libHandler : libHandlers) {
    auto apiFunc = GetOpApiFuncFromLib(libHandler.first, libHandler.second.c_str(), apiName);
    if (apiFunc != nullptr) {
      (void)opapiCache.emplace(std::string(apiName), apiFunc);
      RT_VLOG(VL_OPS) << "Get OpApiFunc [" << apiName << "] from " << libHandler.second;
      return apiFunc;
    }
  }
  RT_VLOG(VL_OPS) << "Dlsym " << apiName << " failed";
  (void)opapiCache.emplace(std::string(apiName), nullptr);
  return nullptr;
}

void AclnnInit() {
  std::lock_guard<std::mutex> lock(initMutex);
  if (isAclnnInit) {
    return;
  }
  static const auto aclnnInit = GET_ACLNN_COMMON_META_FUNC(aclnnInit);
  CHECK_IF_NULL(aclnnInit);
  auto ret = aclnnInit(nullptr);
  CHECK_IF_FAIL(ret == 0);
  isAclnnInit = true;
  RT_VLOG(VL_OPS) << "Aclnn init success";
}

void AclnnFinalize() {
  if (!isAclnnInit) {
    return;
  }
  static const auto aclnnFinalize = GET_ACLNN_COMMON_META_FUNC(aclnnFinalize);
  CHECK_IF_NULL(aclnnFinalize);
  auto ret = aclnnFinalize();
  CHECK_IF_FAIL(ret == 0);
  isAclnnInit = false;
  RT_VLOG(VL_OPS) << "Aclnn finalize success";
}

} // namespace ops
} // namespace fxrt

extern "C" DA_API void FxrtAclnnFinalize() {
  fxrt::ops::AclnnFinalize();
}
