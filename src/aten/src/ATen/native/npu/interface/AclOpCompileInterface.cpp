#include "AclOpCompileInterface.h"
#include "c10/npu/register/FunctionLoader.h"
#include "c10/util/Exception.h"
namespace at {
namespace native {
namespace npu {

#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libacl_op_compiler, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)              \
  GET_FUNCTION(libacl_op_compiler, funcName)

REGISTER_LIBRARY(libacl_op_compiler)
LOAD_FUNCTION(aclopSetCompileFlag)

aclError AclopSetCompileFlag(aclOpCompileFlag flag) {
    typedef aclError(*aclopSetCompileFlagFunc)(aclOpCompileFlag);
  static aclopSetCompileFlagFunc func = nullptr;
  if (func == nullptr) {
    func = (aclopSetCompileFlagFunc)GET_FUNC(aclopSetCompileFlag);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclopSetCompileFlag");
  auto ret = func(flag);
  return ret;
}

} // namespace npu
} // namespace native
} // namespace at