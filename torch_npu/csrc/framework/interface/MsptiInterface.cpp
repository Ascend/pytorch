#include "torch_npu/csrc/framework/interface/MsptiInterface.h"

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"

namespace at_npu {
namespace native {

#undef TORCH_NPU_LOAD_FUNC
#define TORCH_NPU_LOAD_FUNC(funcName) \
  TORCH_NPU_REGISTER_FUNCTION(libmspti, funcName)

#undef TORCH_NPU_GET_FUNC
#define TORCH_NPU_GET_FUNC(funcName) \
  TORCH_NPU_GET_FUNCTION(libmspti, funcName)

TORCH_NPU_REGISTER_LIBRARY(libmspti)
TORCH_NPU_LOAD_FUNC(msptiActivityIsEnabled)

bool MsptiActivityIsEnabled(msptiActivityKind kind)
{
    using MsptiActivityIsEnabledFunc = bool (*)(msptiActivityKind);
    static MsptiActivityIsEnabledFunc func = nullptr;
    static bool noFuncFlag = false;
    if (noFuncFlag) {
        return false;
    }
    if (func == nullptr) {
        func = (MsptiActivityIsEnabledFunc)TORCH_NPU_GET_FUNC(msptiActivityIsEnabled);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func msptiActivityIsEnabled");
            noFuncFlag = true;
            return false;
        }
    }
    return func(kind);
}

} // namespace native
} // namespace at_npu
