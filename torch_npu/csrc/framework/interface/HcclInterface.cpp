#include "torch_npu/csrc/framework/interface/HcclInterface.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

namespace at_npu {
namespace native {
namespace hccl {
#undef TORCH_NPU_LOAD_FUNC
#define TORCH_NPU_LOAD_FUNC(funcName) \
  TORCH_NPU_REGISTER_FUNCTION(libhccl, funcName)

#undef TORCH_NPU_GET_FUNC
#define TORCH_NPU_GET_FUNC(funcName) \
  TORCH_NPU_GET_FUNCTION(libhccl, funcName)

TORCH_NPU_REGISTER_LIBRARY(libhccl)
TORCH_NPU_LOAD_FUNC(HcclSetConfig)


extern HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue) {
    typedef HcclResult (*HcclSetConfigFunc)(HcclConfig config, HcclConfigValue configValue);
    static HcclSetConfigFunc func = nullptr;
    if (func == nullptr)
    {
        func = (HcclSetConfigFunc)TORCH_NPU_GET_FUNC(HcclSetConfig);
    }
    if (func == nullptr) {
        TORCH_NPU_WARN(
            "Failed to find this HcclSetConfig function, get real hccl config, need to upgrade hccl version!");
        return HcclResult::HCCL_SUCCESS;
    }
    return func(config, configValue);
}
} // namespace hccl
} // namespace native
} // namespace at_npu