#include "SkInterface.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

namespace c10_npu {
namespace skapi {
#undef TORCH_NPU_LOAD_FUNC
#define TORCH_NPU_LOAD_FUNC(funcName) \
    TORCH_NPU_REGISTER_FUNCTION(libascendsk, funcName)

#undef TORCH_NPU_GET_FUNC
#define TORCH_NPU_GET_FUNC(funcName)           \
    TORCH_NPU_GET_FUNCTION(libascendsk, funcName)

TORCH_NPU_REGISTER_LIBRARY(libascendsk)
TORCH_NPU_LOAD_FUNC(aclskOptimize)
TORCH_NPU_LOAD_FUNC(aclskScopeBegin)
TORCH_NPU_LOAD_FUNC(aclskScopeEnd)

aclError AclskOptimize(aclmdlRI modelRI, const aclskOptions *options)
{
    typedef aclError (*AclskOptimize)(aclmdlRI, const aclskOptions *);
    static AclskOptimize func = nullptr;
    if (func == nullptr) {
        func = (AclskOptimize) TORCH_NPU_GET_FUNC(aclskOptimize);
    }

    TORCH_CHECK(func, "Failed to find function AclskOptimize", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(modelRI, options);
}

aclError AclskScopeBegin(const char *scopeName, aclrtStream stream)
{
    typedef aclError (*AclskScopeBegin)(const char *, aclrtStream);
    static AclskScopeBegin func = nullptr;
    if (func == nullptr) {
        func = (AclskScopeBegin) TORCH_NPU_GET_FUNC(aclskScopeBegin);
    }

    TORCH_CHECK(func, "Failed to find function aclskScopeBegin", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(scopeName, stream);
}

aclError AclskScopeEnd(const char *scopeName, aclrtStream stream)
{
    typedef aclError (*AclskScopeEnd)(const char *, aclrtStream);
    static AclskScopeEnd func = nullptr;
    if (func == nullptr) {
        func = (AclskScopeEnd) TORCH_NPU_GET_FUNC(aclskScopeEnd);
    }

    TORCH_CHECK(func, "Failed to find function aclskScopeEnd", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(scopeName, stream);
}

} // namespace skapi
} // namespace c10_npu