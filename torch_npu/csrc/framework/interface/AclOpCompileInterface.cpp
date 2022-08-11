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

#include <c10/util/Exception.h>

#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"

namespace at_npu
{
  namespace native
  {

#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libacl_op_compiler, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName) \
  GET_FUNCTION(libacl_op_compiler, funcName)

    REGISTER_LIBRARY(libacl_op_compiler)
    LOAD_FUNCTION(aclopSetCompileFlag)
    LOAD_FUNCTION(aclGenGraphAndDumpForOp)
    LOAD_FUNCTION(aclCreateGraphDumpOpt)
    LOAD_FUNCTION(aclDestroyGraphDumpOpt)

aclError AclopSetCompileFlag(aclOpCompileFlag flag) {
  typedef aclError (*aclopSetCompileFlagFunc)(aclOpCompileFlag);
  static aclopSetCompileFlagFunc func = nullptr;
  if (func == nullptr)
  {
    func = (aclopSetCompileFlagFunc)GET_FUNC(aclopSetCompileFlag);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclopSetCompileFlag");
  auto ret = func(flag);
  return ret;
}

aclError AclGenGraphAndDumpForOp(const char *opType,
    int numInputs, const aclTensorDesc *const inputDesc[], const aclDataBuffer *const inputs[],
    int numOutputs, const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
    const aclopAttr *attr, aclopEngineType engineType, const char *graphDumpPath,
    aclGraphDumpOption* graphdumpOpt) {
  typedef aclError(*AclGenGraphAndDumpForOpFunc)(const char *,int,
      const aclTensorDesc *const [], const aclDataBuffer *const [],
      int, const aclTensorDesc *const [], aclDataBuffer *const [],
      const aclopAttr *, aclopEngineType, const char *, aclGraphDumpOption*);
  static AclGenGraphAndDumpForOpFunc func = nullptr;
  if (func == nullptr) {
    func = (AclGenGraphAndDumpForOpFunc)GET_FUNC(aclGenGraphAndDumpForOp);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclGenGraphAndDumpForOp");
  auto ret = func(opType, numInputs, inputDesc, inputs, numOutputs,
      outputDesc, outputs, attr, engineType, graphDumpPath, graphdumpOpt);
  return ret;
}

aclGraphDumpOption* AclCreateGraphDumpOpt() {
  typedef aclGraphDumpOption*(*AclCreateGraphDumpOptFunc)();
  static AclCreateGraphDumpOptFunc func = nullptr;
  if (func == nullptr) {
    func = (AclCreateGraphDumpOptFunc)GET_FUNC(aclCreateGraphDumpOpt);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclCreateGraphDumpOpt");
  return func();
}

aclError AclDestroyGraphDumpOpt(aclGraphDumpOption* aclGraphDumpOpt) {
  typedef aclError(*AclDestroyGraphDumpOptFunc)(aclGraphDumpOption*);
  static AclDestroyGraphDumpOptFunc func = nullptr;
  if (func == nullptr) {
    func = (AclDestroyGraphDumpOptFunc)GET_FUNC(aclDestroyGraphDumpOpt);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclDestroyGraphDumpOpt");
  return func(aclGraphDumpOpt);
}

  } // namespace native
} // namespace at_npu