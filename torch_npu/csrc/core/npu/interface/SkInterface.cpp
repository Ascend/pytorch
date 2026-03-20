#include "SkInterface.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

namespace c10_npu {
namespace skapi {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
    REGISTER_FUNCTION(libascendsk, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)           \
    GET_FUNCTION(libascendsk, funcName)

REGISTER_LIBRARY(libascendsk)
LOAD_FUNCTION(aclskOptimize)
LOAD_FUNCTION(aclskScopeBegin)
LOAD_FUNCTION(aclskScopeEnd)

aclError AclskOptimize(aclmdlRI modelRI, const aclskOptions *options)
{
    typedef aclError (*AclskOptimize)(aclmdlRI, const aclskOptions *);
    static AclskOptimize func = nullptr;
    if (func == nullptr) {
        func = (AclskOptimize) GET_FUNC(aclskOptimize);
    }

    TORCH_CHECK(func, "Failed to find function AclskOptimize", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(modelRI, options);
}

aclError AclskScopeBegin(const char *scopeName, aclrtStream stream)
{
    typedef aclError (*AclskScopeBegin)(const char *, aclrtStream);
    static AclskScopeBegin func = nullptr;
    if (func == nullptr) {
        func = (AclskScopeBegin) GET_FUNC(aclskScopeBegin);
    }

    TORCH_CHECK(func, "Failed to find function aclskScopeBegin", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(scopeName, stream);
}

aclError AclskScopeEnd(const char *scopeName, aclrtStream stream)
{
    typedef aclError (*AclskScopeEnd)(const char *, aclrtStream);
    static AclskScopeEnd func = nullptr;
    if (func == nullptr) {
        func = (AclskScopeEnd) GET_FUNC(aclskScopeEnd);
    }

    TORCH_CHECK(func, "Failed to find function aclskScopeEnd", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(scopeName, stream);
}

} // namespace skapi
} // namespace c10_npu