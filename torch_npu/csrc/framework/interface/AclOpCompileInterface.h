#ifndef __PLUGIN_NATIVE_NPU_INTERFACE_ACLOPCOMPILE__
#define __PLUGIN_NATIVE_NPU_INTERFACE_ACLOPCOMPILE__
#include <c10/util/Optional.h>
#include "third_party/acl/inc/acl/acl_op_compiler.h"

namespace at_npu {
namespace native {

/**
 * @ingroup AscendCL
 * @brief an interface set compile flag
 *
 * @param flag [IN]     flag: ACL_OPCOMPILE_DEFAULT represent static compile
   while ACL_OPCOMPILE_FUZZ represent dynamic compile
 *
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError AclopSetCompileFlag(aclOpCompileFlag flag);

/**
 * @ingroup AscendCL
 * @brief set compile option
 *
 * @param aclCompileOpt [IN]      compile option
 * @param value [IN]              pointer for the option value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError AclSetCompileopt(aclCompileOpt opt, const char *value);

/**
 * @ingroup AscendCL
 * @brief get compile option value size
 *
 * @param aclCompileOpt [IN]      compile option
 *
 * @retval size of compile option value
 */
ACL_FUNC_VISIBILITY c10::optional<size_t> AclGetCompileoptSize(aclCompileOpt opt);

/**
 * @ingroup AscendCL
 * @brief get compile option
 *
 * @param aclCompileOpt [IN]      compile option
 * @param value [OUT]             pointer for the option value
 * @param length [IN]             length of value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError AclGetCompileopt(aclCompileOpt opt, char *value, size_t length);

/**
 * @ingroup AscendCL
 * @brief dump op graph for AOE
 *
 * @param opType [IN]           op type
 * @param numInputs [IN]        number of inputs
 * @param inputDesc [IN]        pointer to array of input tensor descriptions
 * @param inputs [IN]           pointer to array of input buffers
 * @param numOutputs [IN]       number of outputs
 * @param outputDesc [IN]       pointer to array of output tensor descriptions
 * @param outputs [IN]          pointer to array of outputs buffers
 * @param attr [IN]             pointer to instance of aclopAttr.
 *                              may pass nullptr if the op has no attribute
 * @param engineType [IN]       engine type
 * @param compileFlag [IN]      compile flag
 * @param graphDumpPath [IN]    path to save dump graph of op
 * @param aclGraphDumpOption [IN]  dump graph option
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError AclGenGraphAndDumpForOp(const char *opType,
    int numInputs, const aclTensorDesc *const inputDesc[], const aclDataBuffer *const inputs[],
    int numOutputs, const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
    const aclopAttr *attr, aclopEngineType engineType, const char *graphDumpPath,
    aclGraphDumpOption* graphdumpOpt);

/**
 * @brief create the dump option for AclGenGraphAndDumpForOp API, used for AOE
 * @retval created aclGraphDumpOption
 */
aclGraphDumpOption* AclCreateGraphDumpOpt();

/**
 * @brief destroy the dump option created by aclCreateGraphDumpOpt
 * @param aclGraphDumpOpt [IN]     dump option created by aclCreateGraphDumpOpt
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError AclDestroyGraphDumpOpt(aclGraphDumpOption* aclGraphDumpOpt);

/**
 * @ingroup AscendCL
 * @brief compile and execute op
 *
 * @param opType [IN]           op type
 * @param numInputs [IN]        number of inputs
 * @param inputDesc [IN]        pointer to array of input tensor descriptions
 * @param inputs [IN]           pointer to array of input buffers
 * @param numOutputs [IN]       number of outputs
 * @param outputDesc [IN]       pointer to array of output tensor descriptions
 * @param outputs [IN]          pointer to array of outputs buffers
 * @param attr [IN]             pointer to instance of aclopAttr.
 *                              may pass nullptr if the op has no attribute
 * @param engineType [IN]       engine type
 * @param compileFlag [IN]      compile flag
 * @param opPath [IN]           path of op
 * @param stream [IN]           stream handle
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError AclopCompileAndExecuteV2(const char *opType,
    int numInputs, aclTensorDesc *inputDesc[], aclDataBuffer *inputs[],
    int numOutputs, aclTensorDesc *outputDesc[], aclDataBuffer *outputs[],
    aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag,
    const char *opPath, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief set system param option value in current context
 *
 * @param aclCompileOpt [IN]      system option
 * @param value [IN]              value of system option
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError AclrtCtxSetSysParamOpt(aclSysParamOpt opt, int64_t value);

/**
 * @ingroup AscendCL
 * @brief set system param option value
 *
 * @param aclCompileOpt [IN]      system option
 * @param value [IN]              value of system option
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY  aclError AclrtSetSysParamOpt(aclSysParamOpt opt, int64_t value);

} // namespace native
} // namespace at_npu

#endif // __NATIVE_NPU_INTERFACE_ACLOPCOMPILE__