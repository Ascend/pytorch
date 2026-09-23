#ifndef FXRT_SRC_HARDWARE_ASCEND_ACL_COMPILER_SYMBOL_H_
#define FXRT_SRC_HARDWARE_ASCEND_ACL_COMPILER_SYMBOL_H_
#include <string>
#include "acl/acl_op_compiler.h"
#include "hardware/hardware_abstract/dlopen_macro.h"

namespace fxrt::device::ascend {

ORIGIN_METHOD_WITH_SIMU(
    aclopCompileAndExecute,
    aclError,
    const char*,
    int,
    const aclTensorDesc* const[],
    const aclDataBuffer* const[],
    int,
    const aclTensorDesc* const[],
    aclDataBuffer* const[],
    const aclopAttr*,
    aclopEngineType,
    aclopCompileType,
    const char*,
    aclrtStream);
ORIGIN_METHOD_WITH_SIMU(
    aclopCompileAndExecuteV2,
    aclError,
    const char*,
    int,
    aclTensorDesc*[],
    aclDataBuffer*[],
    int,
    aclTensorDesc*[],
    aclDataBuffer*[],
    aclopAttr*,
    aclopEngineType,
    aclopCompileType,
    const char*,
    aclrtStream);
ORIGIN_METHOD_WITH_SIMU(aclSetCompileopt, aclError, aclCompileOpt, const char*);
ORIGIN_METHOD_WITH_SIMU(aclopSetCompileFlag, aclError, aclOpCompileFlag);
ORIGIN_METHOD_WITH_SIMU(
    aclGenGraphAndDumpForOp,
    aclError,
    const char*,
    int,
    const aclTensorDesc* const[],
    const aclDataBuffer* const[],
    int,
    const aclTensorDesc* const[],
    aclDataBuffer* const[],
    const aclopAttr*,
    aclopEngineType,
    const char*,
    const aclGraphDumpOption*);

void LoadAclOpCompilerApiSymbol(const std::string& ascendPath);
void LoadSimulationAclOpCompilerApi();
} // namespace fxrt::device::ascend

#endif // FXRT_SRC_HARDWARE_ASCEND_ACL_COMPILER_SYMBOL_H_
