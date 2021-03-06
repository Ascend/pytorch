// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __PLUGIN_NATIVE_NPU_INTERFACE_ACLOPCOMPILE__
#define __PLUGIN_NATIVE_NPU_INTERFACE_ACLOPCOMPILE__

#include "third_party/acl/inc/acl/acl_op_compiler.h"

namespace at_npu {
namespace native {

/**
 * @ingroup AscendCL
 * @brief an interface set compile flag
 *
 * @param flag [IN]     flag: ACL_OPCOMPILE_DEFAULT represent static compile while ACL_OPCOMPILE_FUZZ represent dynamic compile
 *
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError AclopSetCompileFlag(aclOpCompileFlag flag);

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


} // namespace native
} // namespace at_npu

#endif // __NATIVE_NPU_INTERFACE_ACLOPCOMPILE__