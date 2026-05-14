#include <c10/util/Optional.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "third_party/acl/inc/acl/acl_base.h"

namespace at_npu
{
  namespace native
  {
#undef TORCH_NPU_LOAD_FUNC
#define TORCH_NPU_LOAD_FUNC(funcName) \
  TORCH_NPU_REGISTER_FUNCTION(libacl_op_compiler, funcName)

#undef TORCH_NPU_GET_FUNC
#define TORCH_NPU_GET_FUNC(funcName) \
  TORCH_NPU_GET_FUNCTION(libacl_op_compiler, funcName)

    TORCH_NPU_REGISTER_LIBRARY(libacl_op_compiler)
    TORCH_NPU_LOAD_FUNC(aclSetCompileopt)
    TORCH_NPU_LOAD_FUNC(aclGetCompileoptSize)
    TORCH_NPU_LOAD_FUNC(aclGetCompileopt)
    TORCH_NPU_LOAD_FUNC(aclGenGraphAndDumpForOp)
    TORCH_NPU_LOAD_FUNC(aclCreateGraphDumpOpt)
    TORCH_NPU_LOAD_FUNC(aclDestroyGraphDumpOpt)
    TORCH_NPU_LOAD_FUNC(aclopCompileAndExecuteV2)
    TORCH_NPU_LOAD_FUNC(aclrtCtxSetSysParamOpt)
    TORCH_NPU_LOAD_FUNC(aclrtSetSysParamOpt)

aclError AclSetCompileopt(aclCompileOpt opt, const char *value)
{
    bool ge_init_disable = c10_npu::option::OptionsManager::CheckGeInitDisable();
    if (ge_init_disable || c10_npu::IsAclnnOnly()) {
        return ACL_ERROR_NONE;
    }
    typedef aclError (*aclSetCompileoptFunc)(aclCompileOpt opt, const char *value);
    static aclSetCompileoptFunc func = nullptr;
    if (func == nullptr) {
        func = (aclSetCompileoptFunc)TORCH_NPU_GET_FUNC(aclSetCompileopt);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclSetCompileopt", OPS_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(opt, value);
    return ret;
}

c10::optional<size_t> AclGetCompileoptSize(aclCompileOpt opt)
{
    typedef aclError (*aclGetCompileoptSizeFunc)(aclCompileOpt opt);
    static aclGetCompileoptSizeFunc func = nullptr;
    if (func == nullptr) {
        func = (aclGetCompileoptSizeFunc)TORCH_NPU_GET_FUNC(aclGetCompileoptSize);
    }
    if (func == nullptr) {
        return c10::nullopt;
    } else {
        return func(opt);
    }
}

aclError AclGetCompileopt(aclCompileOpt opt, char *value, size_t length)
{
    typedef aclError (*aclGetCompileoptFunc)(aclCompileOpt opt, char *value, size_t length);
    static aclGetCompileoptFunc func = nullptr;
    if (func == nullptr) {
        func = (aclGetCompileoptFunc)TORCH_NPU_GET_FUNC(aclGetCompileopt);
    }
    if (func == nullptr) {
        return ACL_ERROR_GE_FAILURE;
    } else {
        return func(opt, value, length);
    }
}

aclError AclGenGraphAndDumpForOp(const char *opType,
    int numInputs, const aclTensorDesc *const inputDesc[], const aclDataBuffer *const inputs[],
    int numOutputs, const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
    const aclopAttr *attr, aclopEngineType engineType, const char *graphDumpPath,
    aclGraphDumpOption* graphdumpOpt)
{
    typedef aclError(*AclGenGraphAndDumpForOpFunc)(const char *, int,
        const aclTensorDesc *const [], const aclDataBuffer *const [],
        int, const aclTensorDesc *const [], aclDataBuffer *const [],
        const aclopAttr *, aclopEngineType, const char *, aclGraphDumpOption*);
    static AclGenGraphAndDumpForOpFunc func = nullptr;
    if (func == nullptr) {
        func = (AclGenGraphAndDumpForOpFunc)TORCH_NPU_GET_FUNC(aclGenGraphAndDumpForOp);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclGenGraphAndDumpForOp", OPS_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(opType, numInputs, inputDesc, inputs, numOutputs,
        outputDesc, outputs, attr, engineType, graphDumpPath, graphdumpOpt);
    return ret;
}

aclGraphDumpOption* AclCreateGraphDumpOpt()
{
    typedef aclGraphDumpOption*(*AclCreateGraphDumpOptFunc)();
    static AclCreateGraphDumpOptFunc func = nullptr;
    if (func == nullptr) {
        func = (AclCreateGraphDumpOptFunc)TORCH_NPU_GET_FUNC(aclCreateGraphDumpOpt);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclCreateGraphDumpOpt", OPS_ERROR(ErrCode::NOT_FOUND));
    return func();
}

aclError AclDestroyGraphDumpOpt(aclGraphDumpOption* aclGraphDumpOpt)
{
    typedef aclError(*AclDestroyGraphDumpOptFunc)(aclGraphDumpOption*);
    static AclDestroyGraphDumpOptFunc func = nullptr;
    if (func == nullptr) {
        func = (AclDestroyGraphDumpOptFunc)TORCH_NPU_GET_FUNC(aclDestroyGraphDumpOpt);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclDestroyGraphDumpOpt", OPS_ERROR(ErrCode::NOT_FOUND));
    return func(aclGraphDumpOpt);
}

aclError AclopCompileAndExecuteV2(const char *opType,
    int numInputs, aclTensorDesc *inputDesc[], aclDataBuffer *inputs[],
    int numOutputs, aclTensorDesc *outputDesc[], aclDataBuffer *outputs[],
    aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag,
    const char *opPath, aclrtStream stream)
{
    typedef aclError(*AclopCompileAndExecuteV2Func)(const char *,
        int, aclTensorDesc * [], aclDataBuffer * [],
        int, aclTensorDesc * [], aclDataBuffer * [],
        aclopAttr *, aclopEngineType, aclopCompileType,
        const char *, aclrtStream);
    static AclopCompileAndExecuteV2Func func = nullptr;
    if (func == nullptr) {
        func = (AclopCompileAndExecuteV2Func)TORCH_NPU_GET_FUNC(aclopCompileAndExecuteV2);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclopCompileAndExecuteV2", OPS_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(opType, numInputs, inputDesc, inputs, numOutputs,
        outputDesc, outputs, attr, engineType, compileFlag, opPath, stream);
    return ret;
}

aclError AclrtCtxSetSysParamOpt(aclSysParamOpt opt, int64_t value)
{
    typedef aclError (*AclrtCtxSetSysParamOptFunc)(aclSysParamOpt opt, int64_t value);
    static AclrtCtxSetSysParamOptFunc func = nullptr;
    if (func == nullptr) {
        func = (AclrtCtxSetSysParamOptFunc)TORCH_NPU_GET_FUNC(aclrtCtxSetSysParamOpt);
    }
    if (func == nullptr) {
        TORCH_WARN("Failed to find this aclrtCtxSetSysParamOpt function!");
        return ACL_ERROR_NONE;
    }
    auto ret = func(opt, value);
    return ret;
}

aclError AclrtSetSysParamOpt(aclSysParamOpt opt, int64_t value)
{
    typedef aclError (*AclrtSetSysParamOptFunc)(aclSysParamOpt opt, int64_t value);
    static AclrtSetSysParamOptFunc func = nullptr;
    if (func == nullptr) {
        func = (AclrtSetSysParamOptFunc)TORCH_NPU_GET_FUNC(aclrtSetSysParamOpt);
    }
    if (func == nullptr) {
        TORCH_WARN("Failed to find this aclrtSetSysParamOpt function!");
        return ACL_ERROR_NONE;
    }
    auto ret = func(opt, value);
    return ret;
}

#undef LOAD_OPBASE_FUNCTION
#define LOAD_OPBASE_FUNCTION(funcName) \
  TORCH_NPU_REGISTER_FUNCTION(libnnopbase, funcName)

#undef GET_OPBASE_FUNC
#define GET_OPBASE_FUNC(funcName) \
  TORCH_NPU_GET_FUNCTION(libnnopbase, funcName)

TORCH_NPU_REGISTER_LIBRARY(libnnopbase)
LOAD_OPBASE_FUNCTION(aclDestroyAclOpExecutor)

aclError AclDestroyAclOpExecutor(aclOpExecutor *executor)
{
    typedef aclError (*aclDestroyAclOpExecutorFunc)(aclOpExecutor *executor);
    static aclDestroyAclOpExecutorFunc func = nullptr;
    if (func == nullptr) {
        func = (aclDestroyAclOpExecutorFunc)GET_OPBASE_FUNC(aclDestroyAclOpExecutor);
    }
    TORCH_CHECK(func, "Failed to find function  ", "aclDestroyAclOpExecutor", OPS_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(executor);
    return ret;
}

  } // namespace native
} // namespace at_npu