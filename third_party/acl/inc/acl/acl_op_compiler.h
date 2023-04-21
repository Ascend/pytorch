/**
* @file acl_op_compiler.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef INC_EXTERNAL_ACL_ACL_OP_COMPILER_H_
#define INC_EXTERNAL_ACL_ACL_OP_COMPILER_H_

#include "acl_base.h"
#include "acl_op.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum aclCompileType {
    ACL_COMPILE_SYS,
    ACL_COMPILE_UNREGISTERED
} aclopCompileType;

typedef enum {
    ACL_PRECISION_MODE,
    ACL_AICORE_NUM,
    ACL_AUTO_TUNE_MODE,
    ACL_OP_SELECT_IMPL_MODE,
    ACL_OPTYPELIST_FOR_IMPLMODE,
    ACL_OP_DEBUG_LEVEL,
    ACL_DEBUG_DIR,
    ACL_OP_COMPILER_CACHE_MODE,
    ACL_OP_COMPILER_CACHE_DIR,
    ACL_OP_PERFORMANCE_MODE,
    ACL_OP_JIT_COMPILE,
    ACL_OP_DETERMINISTIC,
    ACL_CUSTOMIZE_DTYPES,
    ACL_OP_PRECISION_MODE,
    ACL_ALLOW_HF32
} aclCompileOpt;

typedef enum aclCompileFlag {
    ACL_OP_COMPILE_DEFAULT,
    ACL_OP_COMPILE_FUZZ
} aclOpCompileFlag;

typedef struct aclGraphDumpOption aclGraphDumpOption;

/**
 * @ingroup AscendCL
 * @brief compile op
 *
 * @param opType [IN]           op type
 * @param numInputs [IN]        number of inputs
 * @param inputDesc [IN]        pointer to array of input tensor descriptions
 * @param numOutputs [IN]       number of outputs
 * @param outputDesc [IN]       pointer to array of output tensor descriptions
 * @param attr [IN]           pointer to instance of aclopAttr.
 *                              may pass nullptr if the op has no attribute
 * @param engineType [IN]       engine type
 * @param compileFlag [IN]      compile flag
 * @param opPath [IN]           path of op
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopCompile(const char *opType,
                                          int numInputs,
                                          const aclTensorDesc *const inputDesc[],
                                          int numOutputs,
                                          const aclTensorDesc *const outputDesc[],
                                          const aclopAttr *attr,
                                          aclopEngineType engineType,
                                          aclopCompileType compileFlag,
                                          const char *opPath);

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
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopCompileAndExecute(const char *opType,
    int numInputs, const aclTensorDesc *const inputDesc[], const aclDataBuffer *const inputs[],
    int numOutputs, const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
    const aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag,
    const char *opPath, aclrtStream stream);


/**
 * @ingroup AscendCL
 * @brief compile and execute op
 *
 * @param opType [IN]           op type
 * @param numInputs [IN]        number of inputs
 * @param inputDesc [IN]        pointer to array of input tensor descriptions
 * @param inputs [IN]           pointer to array of input buffers
 * @param numOutputs [IN]       number of outputs
 * @param outputDesc [IN|OUT]   pointer to array of output tensor descriptions
 * @param outputs [IN]          pointer to array of outputs buffers
 * @param attr [IN]             pointer to instance of aclopAttr.
 *                              may pass nullptr if the op has no attribute
 * @param engineType [IN]       engine type
 * @param compileFlag [IN]      compile flag
 * @param opPath [IN]           path of op
 * @param stream [IN]           stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopCompileAndExecuteV2(const char *opType,
    int numInputs, aclTensorDesc *inputDesc[], aclDataBuffer *inputs[],
    int numOutputs, aclTensorDesc *outputDesc[], aclDataBuffer *outputs[],
    aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag,
    const char *opPath, aclrtStream stream);

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
ACL_FUNC_VISIBILITY aclError aclSetCompileopt(aclCompileOpt opt, const char *value);

/**
 * @ingroup AscendCL
 * @brief get compile option value size
 *
 * @param aclCompileOpt [IN]      compile option
 *
 * @retval size of compile option value
 */
ACL_FUNC_VISIBILITY size_t aclGetCompileoptSize(aclCompileOpt opt);

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
ACL_FUNC_VISIBILITY aclError aclGetCompileopt(aclCompileOpt opt, char *value, size_t length);

/**
 * @ingroup AscendCL
 * @brief set compile flag
 *
 * @param flag [IN]    compile flag, ACL_OP_COMPILE_DEFAULT means compile with default mode
 *                     ACL_OP_COMPILE_FUZZ means compile with fuzz mode
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetCompileFlag(aclOpCompileFlag flag);

/**
 * @ingroup AscendCL
 * @brief generate graph and dump
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
 * @param graphDumpPath [IN]    dump path, if the suffix is ".txt", it means file path, else it means directory path
 * @param graphDumpOpt [IN]     dump option, nullptr is supported
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclGenGraphAndDumpForOp(const char *opType,
    int numInputs, const aclTensorDesc *const inputDesc[], const aclDataBuffer *const inputs[],
    int numOutputs, const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
    const aclopAttr *attr, aclopEngineType engineType,
    const char *graphDumpPath, const aclGraphDumpOption *graphDumpOpt);

/**
 * @ingroup AscendCL
 * @brief Create the graph dump option
 *
 * @retval null for failed
 * @retval OtherValues success
 *
 * @see aclDestroyGraphDumpOpt
 */
ACL_FUNC_VISIBILITY aclGraphDumpOption *aclCreateGraphDumpOpt();

/**
 * @ingroup AscendCL
 * @brief Destroy graph dump option
 *
 * @param graphDumpOpt [IN]  pointer to the graph dump option
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclCreateGraphDumpOpt
 */
ACL_FUNC_VISIBILITY aclError aclDestroyGraphDumpOpt(const aclGraphDumpOption *graphDumpOpt);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_OP_COMPILER_H_
