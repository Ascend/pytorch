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
LOAD_FUNCTION(aclnnSilentCheckV2)

bool IsExistAclnnSilentCheck()
{
    const static bool isExist = []() -> bool {
        static auto func = GET_FUNC(aclnnSilentCheck);
        return func != nullptr;
    }();
    return isExist;
}

bool IsExistAclnnSilentCheckV2()
{
    const static bool isExistV2 = []() -> bool {
        static auto func = GET_FUNC(aclnnSilentCheckV2);
        return func != nullptr;
    }();
    return isExistV2;
}

} // namespace opapi
} // namespace c10_npu
