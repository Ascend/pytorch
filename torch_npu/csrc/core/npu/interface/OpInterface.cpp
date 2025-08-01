#include "OpInterface.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "third_party/op-plugin/op_plugin/utils/op_api_common.h"

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
        static auto func = GetOpApiFuncAddr("aclnnSilentCheck");
        return func != nullptr;
    }();
    return isExist;
}

} // namespace opapi
} // namespace c10_npu
