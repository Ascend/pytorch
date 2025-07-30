#include "OpInterface.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace c10_npu {

namespace opapi {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libopapi, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)           \
  GET_FUNCTION(libopapi, funcName)

REGISTER_LIBRARY(libopapi)
LOAD_FUNCTION(aclnnSilentCheck)
LOAD_FUNCTION(aclnnSilentCheckV2)
LOAD_FUNCTION(aclnnReselectStaticKernel)

bool IsExistAclnnSilentCheck()
{
    const static bool isExist = []() -> bool {
        static auto func = GET_FUNC(aclnnSilentCheck);
        return func != nullptr;
    }();
    return isExist;
}

aclnnStatus ReselectStaticKernel()
{
    typedef aclnnStatus (*AclnnApiFunc)();
    static AclnnApiFunc aclnnReselectStaticKernelFunc = nullptr;
    if (aclnnReselectStaticKernelFunc == nullptr) {
        aclnnReselectStaticKernelFunc = (AclnnApiFunc)GET_FUNC(aclnnReselectStaticKernel);
    }
    TORCH_CHECK(aclnnReselectStaticKernelFunc,
        "Failed to find function ",
        "aclnnReselectStaticKernel",
        PTA_ERROR(ErrCode::NOT_FOUND));
    auto ret = aclnnReselectStaticKernelFunc();
    return ret;
}

} // namespace opapi
} // namespace c10_npu
