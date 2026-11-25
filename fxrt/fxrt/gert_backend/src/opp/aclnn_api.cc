#include "opp/aclnn_api.h"

#include <dlfcn.h>

#include <array>
#include <cstring>
#include <utility>

namespace fx_converter {
namespace aclnn_api {

fxrt::ops::_aclCreateTensorFuncPtr aclCreateTensor = nullptr;
fxrt::ops::_aclCreateScalarFuncPtr aclCreateScalar = nullptr;
fxrt::ops::_aclCreateIntArrayFuncPtr aclCreateIntArray = nullptr;
fxrt::ops::_aclCreateTensorListFuncPtr aclCreateTensorList = nullptr;
fxrt::ops::_aclDestroyTensorFuncPtr aclDestroyTensor = nullptr;
fxrt::ops::_aclDestroyScalarFuncPtr aclDestroyScalar = nullptr;
fxrt::ops::_aclDestroyIntArrayFuncPtr aclDestroyIntArray = nullptr;
fxrt::ops::_aclDestroyTensorListFuncPtr aclDestroyTensorList = nullptr;
void *aclnnAddGetWorkspaceSize = nullptr;
void *aclnnAdd = nullptr;
void *aclnnAddmmGetWorkspaceSize = nullptr;
void *aclnnAddmm = nullptr;
void *aclnnAmaxGetWorkspaceSize = nullptr;
void *aclnnAmax = nullptr;
void *aclnnBaddbmmGetWorkspaceSize = nullptr;
void *aclnnBaddbmm = nullptr;
void *aclnnBatchMatMulGetWorkspaceSize = nullptr;
void *aclnnBatchMatMul = nullptr;
void *aclnnBitwiseNotGetWorkspaceSize = nullptr;
void *aclnnBitwiseNot = nullptr;
void *aclnnCatGetWorkspaceSize = nullptr;
void *aclnnCat = nullptr;
void *aclnnConvolutionGetWorkspaceSize = nullptr;
void *aclnnConvolution = nullptr;
void *aclnnFlashAttentionScoreGetWorkspaceSize = nullptr;
void *aclnnFlashAttentionScore = nullptr;
void *aclnnGatherGetWorkspaceSize = nullptr;
void *aclnnGather = nullptr;
void *aclnnIndexGetWorkspaceSize = nullptr;
void *aclnnIndex = nullptr;
void *aclnnInplaceCopyGetWorkspaceSize = nullptr;
void *aclnnInplaceCopy = nullptr;
void *aclnnInplaceFillScalarGetWorkspaceSize = nullptr;
void *aclnnInplaceFillScalar = nullptr;
void *aclnnMatmulGetWorkspaceSize = nullptr;
void *aclnnMatmul = nullptr;
void *aclnnMmGetWorkspaceSize = nullptr;
void *aclnnMm = nullptr;
void *aclnnReduceSumGetWorkspaceSize = nullptr;
void *aclnnReduceSum = nullptr;
void *aclnnSWhereGetWorkspaceSize = nullptr;
void *aclnnSWhere = nullptr;

namespace {
struct CommonApi {
  void *handle = nullptr;
  decltype(&FxrtAclnnEnsureInitializedV1) initialize = nullptr;
  decltype(&FxrtAclnnResolveV1) resolve = nullptr;
  std::string error;
};

const CommonApi &GetCommonApi() {
  static const CommonApi api = [] {
    CommonApi result;
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(&GetCommonApi), &info) == 0 || info.dli_fname == nullptr) {
      result.error = "Cannot locate the converter SO for the FXRT ACLNN loader";
      return result;
    }
    const std::string converter_path(info.dli_fname);
    const auto slash = converter_path.find_last_of('/');
    if (slash == std::string::npos) {
      result.error = "Converter SO path has no package directory: " + converter_path;
      return result;
    }
    const std::string path = converter_path.substr(0, slash) + "/../../lib/libops_ascend_aclnn_common.so";
    result.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (result.handle == nullptr) {
      const char *detail = dlerror();
      result.error = "Cannot load " + path + ": " + (detail == nullptr ? "unknown dlopen error" : detail);
      return result;
    }
    result.initialize = reinterpret_cast<decltype(result.initialize)>(
        dlsym(result.handle, "FxrtAclnnEnsureInitializedV1"));
    result.resolve = reinterpret_cast<decltype(result.resolve)>(dlsym(result.handle, "FxrtAclnnResolveV1"));
    if (result.initialize == nullptr || result.resolve == nullptr) {
      result.error = "FXRT ACLNN V1 interface missing from " + path + "; install matching FXRT/converter artifacts";
    }
    // The existing FXRT loader owns CANN handles. Retain the common SO for the
    // lifetime of the borrowed addresses; do not finalize or unload it here.
    return result;
  }();
  return api;
}

template <typename... Functions>
bool ResolveGroup(std::string *error, const char *const (&names)[sizeof...(Functions)], Functions *...targets) {
  const auto &api = GetCommonApi();
  if (!api.error.empty()) {
    *error = api.error;
    return false;
  }
  std::array<void *, sizeof...(Functions)> addresses{};
  char message[512]{};
  if (api.resolve(names, addresses.size(), addresses.data(), message, sizeof(message)) != 0) {
    *error = message;
    return false;
  }
  size_t index = 0U;
  ((*targets = reinterpret_cast<Functions>(addresses[index++])), ...);
  return true;
}

bool EnsureMeta(std::string *error) {
  // Called under opp_binding.cc's mutex only. Once published, no pointer is
  // written again while previously compiled graphs are executing.
  static bool initialized = false;
  if (initialized) {
    return true;
  }
  const auto &api = GetCommonApi();
  if (!api.error.empty()) {
    *error = api.error;
    return false;
  }
  char message[512]{};
  if (api.initialize(message, sizeof(message)) != 0) {
    *error = "FXRT ACLNN initialization failed: " + std::string(message);
    return false;
  }
  initialized = ResolveGroup(error,
      {"aclCreateTensor", "aclCreateScalar", "aclCreateIntArray", "aclCreateTensorList", "aclDestroyTensor", "aclDestroyScalar", "aclDestroyIntArray", "aclDestroyTensorList"},
      &aclCreateTensor, &aclCreateScalar, &aclCreateIntArray, &aclCreateTensorList, &aclDestroyTensor, &aclDestroyScalar, &aclDestroyIntArray, &aclDestroyTensorList);
  return initialized;
}

struct OpSymbols {
  const char *opp_type;
  bool (*resolve)(std::string *);
};

#define FXC_ACLNN_PAIR(opp_type, name) \
  {opp_type, [](std::string *error) { \
    return ResolveGroup(error, {#name "GetWorkspaceSize", #name}, &name##GetWorkspaceSize, &name); \
  }}

const OpSymbols kOpSymbols[] = {
    FXC_ACLNN_PAIR("Add", aclnnAdd),
    FXC_ACLNN_PAIR("Invert", aclnnBitwiseNot),
    FXC_ACLNN_PAIR("GatherElements", aclnnGather),
    FXC_ACLNN_PAIR("Index", aclnnIndex),
    FXC_ACLNN_PAIR("SelectV2", aclnnSWhere),
    FXC_ACLNN_PAIR("ReduceSum", aclnnReduceSum),
    FXC_ACLNN_PAIR("ReduceMax", aclnnAmax),
    FXC_ACLNN_PAIR("BatchMatMul", aclnnBatchMatMul),
    FXC_ACLNN_PAIR("MatMulV2", aclnnAddmm),
    FXC_ACLNN_PAIR("Concat", aclnnCat),
    FXC_ACLNN_PAIR("Fills", aclnnInplaceFillScalar),
    FXC_ACLNN_PAIR("Assign", aclnnInplaceCopy),
    FXC_ACLNN_PAIR("Conv2D", aclnnConvolution),
    FXC_ACLNN_PAIR("BatchMatMulV2", aclnnBaddbmm),
    FXC_ACLNN_PAIR("FlashAttentionScore", aclnnFlashAttentionScore),
    {"MatMul", [](std::string *error) {
       return ResolveGroup(error,
           {"aclnnMmGetWorkspaceSize", "aclnnMm", "aclnnMatmulGetWorkspaceSize", "aclnnMatmul"},
           &aclnnMmGetWorkspaceSize, &aclnnMm, &aclnnMatmulGetWorkspaceSize, &aclnnMatmul);
     }},
};
#undef FXC_ACLNN_PAIR
}  // namespace

bool EnsureForOp(const char *opp_type, std::string *error) {
  std::string ignored;
  if (error == nullptr) {
    error = &ignored;
  }
  if (opp_type == nullptr) {
    *error = "ACLNN OPP type must not be null";
    return false;
  }
  for (const auto &symbols : kOpSymbols) {
    if (std::strcmp(opp_type, symbols.opp_type) == 0) {
      return EnsureMeta(error) && symbols.resolve(error);
    }
  }
  // Other OPP implementations own their own dependencies.
  return true;
}

}  // namespace aclnn_api
}  // namespace fx_converter
