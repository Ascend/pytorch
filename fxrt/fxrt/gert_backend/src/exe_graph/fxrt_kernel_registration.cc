#include "exe_graph/fxrt_kernel_registration.h"

#include <dlfcn.h>

#include <unordered_map>

namespace fx_converter {
namespace {

#define FXRT_KERNEL_ENTRY(name) \
  {#name, {#name, "fxrt_" #name "_Run", nullptr}}
#define FXRT_KERNEL_ENTRY_WITH_OUTPUTS(name) \
  {#name, {#name, "fxrt_" #name "_Run", "fxrt_" #name "_CreateOutputs"}}

const std::unordered_map<std::string, FxrtKernelRegistration> &Registrations() {
  static const std::unordered_map<std::string, FxrtKernelRegistration> registrations = {
      FXRT_KERNEL_ENTRY_WITH_OUTPUTS(FxFusedInferShape),
      FXRT_KERNEL_ENTRY(BindSymbolEnv),
      FXRT_KERNEL_ENTRY(FxDeserializeInferFunc),
      FXRT_KERNEL_ENTRY(FxGetStorageOffset),
      FXRT_KERNEL_ENTRY_WITH_OUTPUTS(FxAiCoreTiling),
      FXRT_KERNEL_ENTRY(FxAiCoreLaunch),
      FXRT_KERNEL_ENTRY_WITH_OUTPUTS(FxBuildStridedTensor),
      FXRT_KERNEL_ENTRY_WITH_OUTPUTS(FxBuildLayoutShape),
      FXRT_KERNEL_ENTRY_WITH_OUTPUTS(FxBuildStorageCapacityShape),
      FXRT_KERNEL_ENTRY(FxGetExternalAllocators),
      FXRT_KERNEL_ENTRY(InductorAutofuseInit),
      FXRT_KERNEL_ENTRY(InductorAutofuseFinalize),
      FXRT_KERNEL_ENTRY_WITH_OUTPUTS(InductorAutofuseTiling),
      FXRT_KERNEL_ENTRY(InductorAutofuseLaunch),
      FXRT_KERNEL_ENTRY(FxExecuteOpPrepare),
      FXRT_KERNEL_ENTRY(FxExecuteOpLaunch),
      FXRT_KERNEL_ENTRY_WITH_OUTPUTS(FxBuildTensor),
      FXRT_KERNEL_ENTRY_WITH_OUTPUTS(FxBuildRefTensor),
      FXRT_KERNEL_ENTRY(FxEnsureTensorAtOutMemory),
      FXRT_KERNEL_ENTRY_WITH_OUTPUTS(FxTorchFallback),
      FXRT_KERNEL_ENTRY_WITH_OUTPUTS(FxTorchFallbackWithInferShape),
      FXRT_KERNEL_ENTRY_WITH_OUTPUTS(FxContiguousView),
      FXRT_KERNEL_ENTRY_WITH_OUTPUTS(FxStorageBaseView),
      FXRT_KERNEL_ENTRY(FxDeserializeConstBytePtr),
      FXRT_KERNEL_ENTRY(FxDeserializeConstBool),
      FXRT_KERNEL_ENTRY(FxDeserializeConstInt32),
      FXRT_KERNEL_ENTRY(FxDeserializeConstInt64),
      FXRT_KERNEL_ENTRY(FxDeserializeConstUInt32),
      FXRT_KERNEL_ENTRY(FxDeserializeConstUInt64),
      FXRT_KERNEL_ENTRY(FxDeserializeConstSizeT),
      FXRT_KERNEL_ENTRY(FxDeserializeConstGeDataType),
      FXRT_KERNEL_ENTRY(FxDeserializeConstTensorPlacement),
      FXRT_KERNEL_ENTRY(FxDeserializeConstCoreTypeIndex),
      FXRT_KERNEL_ENTRY(FxDeserializeConstVoidPtr),
      FXRT_KERNEL_ENTRY(FxDeserializeConstKernelFuncPtr),
      FXRT_KERNEL_ENTRY(FxDeserializeConstRtStreamPtr),
      FXRT_KERNEL_ENTRY(FxDeserializeConstStorageShape),
      FXRT_KERNEL_ENTRY(FxDeserializeConstTensor),
      FXRT_KERNEL_ENTRY(FxDeserializeConstGertTensorData),
      FXRT_KERNEL_ENTRY(FxDeserializeConstTensorData),
      FXRT_KERNEL_ENTRY(FxDeserializeConstOpExecuteOptions),
      FXRT_KERNEL_ENTRY(FxDeserializeConstContinuousVectorSizeT),
      FXRT_KERNEL_ENTRY(FxDeserializeConstContinuousVectorInt64),
      FXRT_KERNEL_ENTRY(FxDeserializeConstContinuousVectorVectorInt32),
      FXRT_KERNEL_ENTRY(FxDeserializeConstTaskConfig),
      FXRT_KERNEL_ENTRY(FxDeserializeConstDfxExeArg),
      FXRT_KERNEL_ENTRY(FxDeserializeConstComputeNodeDesc),
      FXRT_KERNEL_ENTRY(FxDeserializeConstArgsInfosDesc),
      FXRT_KERNEL_ENTRY(FxDeserializeConstLaunchKernelAttrs),
      FXRT_KERNEL_ENTRY(FxDeserializeConstLaunchKernelConfig),
      FXRT_KERNEL_ENTRY(FxDeserializeConstMixCoreArgs),
      FXRT_KERNEL_ENTRY(FxDeserializeConstAllocatorsPtr),
      FXRT_KERNEL_ENTRY(FxDeserializeConstTensorDataVectorPtr),
      FXRT_KERNEL_ENTRY(FxDeserializeConstTilingFwkDataPtr),
  };
  return registrations;
}

#undef FXRT_KERNEL_ENTRY
#undef FXRT_KERNEL_ENTRY_WITH_OUTPUTS

}  // namespace

const FxrtKernelRegistration *FindFxrtKernelRegistration(const std::string &kernel_name) {
  const auto &registrations = Registrations();
  const auto iter = registrations.find(kernel_name);
  return iter == registrations.end() ? nullptr : &iter->second;
}

const std::string &GetFxrtKernelModulePath() {
  static const std::string path = []() {
    static const char module_anchor = 0;
    Dl_info info{};
    if (dladdr(&module_anchor, &info) == 0 || info.dli_fname == nullptr) {
      return std::string{};
    }
    return std::string(info.dli_fname);
  }();
  return path;
}

}  // namespace fx_converter
