#include "OpInterface.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace c10_npu {

namespace opapi {
#undef TORCH_NPU_LOAD_FUNC
#define TORCH_NPU_LOAD_FUNC(funcName) \
  TORCH_NPU_REGISTER_FUNCTION(libopapi, funcName)

#undef TORCH_NPU_GET_FUNC
#define TORCH_NPU_GET_FUNC(funcName)           \
  TORCH_NPU_GET_FUNCTION(libopapi, funcName)

TORCH_NPU_REGISTER_LIBRARY(libopapi)
TORCH_NPU_LOAD_FUNC(aclnnSilentCheck)
TORCH_NPU_LOAD_FUNC(aclnnSilentCheckV2)
TORCH_NPU_LOAD_FUNC(aclnnReselectStaticKernel)

bool IsExistAclnnSilentCheck()
{
    const static bool isExist = []() -> bool {
        static auto func = TORCH_NPU_GET_FUNC(aclnnSilentCheck);
        return func != nullptr;
    }();
    return isExist;
}

aclnnStatus ReselectStaticKernel()
{
    typedef aclnnStatus (*AclnnApiFunc)();
    static AclnnApiFunc aclnnReselectStaticKernelFunc = nullptr;
    if (aclnnReselectStaticKernelFunc == nullptr) {
        aclnnReselectStaticKernelFunc = (AclnnApiFunc)TORCH_NPU_GET_FUNC(aclnnReselectStaticKernel);
    }
    TORCH_CHECK(aclnnReselectStaticKernelFunc, "Failed to find function ", "aclnnReselectStaticKernel", PTA_ERROR(ErrCode::NOT_FOUND));
    auto ret = aclnnReselectStaticKernelFunc();
    return ret;
}

} // namespace opapi
} // namespace c10_npu
