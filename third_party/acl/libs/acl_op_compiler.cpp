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
#include "acl/acl_op_compiler.h"

aclError aclopCompile(
    const char* opType,
    int numInputs,
    const aclTensorDesc* const inputDesc[],
    int numOutputs,
    const aclTensorDesc* const outputDesc[],
    const aclopAttr* attr,
    aclopEngineType engineType,
    aclopCompileType compileFlag,
    const char* opPath) {
  return 0;
}

aclError aclopCompileAndExecute(
    const char* opType,
    int numInputs,
    const aclTensorDesc* const inputDesc[],
    const aclDataBuffer* const inputs[],
    int numOutputs,
    const aclTensorDesc* const outputDesc[],
    aclDataBuffer* const outputs[],
    const aclopAttr* attr,
    aclopEngineType engineType,
    aclopCompileType compileFlag,
    const char* opPath,
    aclrtStream stream) {
  return 0;
}

aclError aclopCompileAndExecuteV2(
    const char* opType,
    int numInputs,
    aclTensorDesc *inputDesc[],
    aclDataBuffer *inputs[],
    int numOutputs,
    aclTensorDesc *outputDesc[],
    aclDataBuffer *outputs[],
    aclopAttr* attr,
    aclopEngineType engineType,
    aclopCompileType compileFlag,
    const char* opPath,
    aclrtStream stream) {
  return 0;
}

void GeGeneratorFinalize() {
  return;
}

aclError aclopExecuteV2(
    const char* opType,
    int numInputs,
    aclTensorDesc* inputDesc[],
    aclDataBuffer* inputs[],
    int numOutputs,
    aclTensorDesc* outputDesc[],
    aclDataBuffer* outputs[],
    aclopAttr* attr,
    aclrtStream stream) {
  return 0;
}

aclError aclopSetAttrListListInt(
    aclopAttr* attr,
    const char* attrName,
    int numLists,
    const int* numValues,
    const int64_t* const values[]) {
  return 0;
}

aclError aclSetTensorConst(
    aclTensorDesc* desc,
    void* dataBuffer,
    size_t length) {
  return 0;
}

aclError aclSetCompileopt(
    aclCompileOpt opt,
    const char* value) {
  return 0;
}

aclError aclGenGraphAndDumpForOp(
    const char *opType,
    int numInputs,
    const aclTensorDesc *const inputDesc[],
    const aclDataBuffer *const inputs[],
    int numOutputs,
    const aclTensorDesc *const outputDesc[],
    aclDataBuffer *const outputs[],
    const aclopAttr *attr,
    aclopEngineType engineType,
    const char *graphDumpPath,
    aclGraphDumpOption* graphdumpOpt) {
  return 0;
}

aclGraphDumpOption* aclCreateGraphDumpOpt() {
  return NULL;
}

aclError aclDestroyGraphDumpOpt(aclGraphDumpOption* aclGraphDumpOpt) {
  return 0;
}