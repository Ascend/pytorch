#include "OpInterface.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

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

bool IsExistAclnnSilentCheck()
{
    const static bool isExist = []() -> bool {
        static auto func = GET_FUNC(aclnnSilentCheck);
        return func != nullptr;
    }();
    return isExist;
}

} // namespace acl
} // namespace c10
