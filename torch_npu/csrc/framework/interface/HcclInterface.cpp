#include "torch_npu/csrc/framework/interface/HcclInterface.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

namespace at_npu {
namespace native {
namespace hccl {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libhccl, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName) \
  GET_FUNCTION(libhccl, funcName)

REGISTER_LIBRARY(libhccl)
LOAD_FUNCTION(HcclSetConfig)


extern HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue) {
    typedef HcclResult (*HcclSetConfigFunc)(HcclConfig config, HcclConfigValue configValue);
    static HcclSetConfigFunc func = nullptr;
    if (func == nullptr)
    {
        func = (HcclSetConfigFunc)GET_FUNC(HcclSetConfig);
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