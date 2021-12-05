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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/frame/OpDynamicParamMaker.h"
#include <third_party/acl/inc/acl/acl_base.h>
#include "c10/npu/NPUStream.h"
#include "ATen/native/npu/interface/AclOpCompileInterface.h"

namespace at {
namespace native {
namespace npu {

// the member in AclExecParam is create by :
// aclCreateDataBuffer and aclCreateTensorDesc
// so aclDestroyTensorDesc and aclDestroyDataBuffer should be called when dtr
// aclopDestroyAttr

void OpDynamicCommandImpl::SetDynamicName(string &name) {
  opDynamicNames.push_back(name);
}

void OpDynamicCommandImpl::AddDynamicKey(string dynamicKey) {
  dynamicKeys.back() += dynamicKey;
}

const string& OpDynamicCommandImpl::GetDynamicName() {
  return opDynamicNames.back();
}

void OpDynamicCommandImpl::AddDynamicInput(
    const aclTensorDesc* desc,
    const aclDataBuffer* buffer,
    int64_t dim,
    aclFormat format) {
  execDynamicParam.inDynamicDesc.emplace_back(desc);
  execDynamicParam.inDynamicBuffer.emplace_back(buffer);
  execDynamicParam.inDynamicDims.emplace_back(dim);
  execDynamicParam.inDynamicFormats.emplace_back(format);
}

void OpDynamicCommandImpl::AddDynamicCompileInputDesc(const aclTensorDesc* desc){
  execDynamicParam.inDynamicCompileDesc.emplace_back(desc);
}
  
void OpDynamicCommandImpl::AddDynamicOutputDesc(const aclTensorDesc* desc) {
  execDynamicParam.outDynamicCompileDesc.emplace_back(desc);
}

void OpDynamicCommandImpl::AddDynamicOutput(
    const aclTensorDesc* desc,
    aclDataBuffer* buffer,
    int64_t dim,
    aclFormat format) {
  
  execDynamicParam.outDynamicDesc.emplace_back(desc);
  execDynamicParam.outDynamicBuffer.emplace_back(buffer);
  execDynamicParam.outDynamicDims.emplace_back(dim);
  execDynamicParam.outDynamicFormats.emplace_back(format);
}

// export op execute params
void OpDynamicCommandImpl::ExportDynamicParams(ExecuteParas& params) {
  InitDynamicAttr();
  params.dynamicCompileAttr = execDynamicParam.dynamicCompileAttrs.back();
  params.dynamicRunAttr = execDynamicParam.dynamicRunAttrs.back();
  params.opDynamicType = opDynamicNames.back();

  int inputDynamicNum = execDynamicParam.inDynamicDesc.size() - execDynamicParam.inDynamicOffset.back();
  int outputDynamicNum = execDynamicParam.outDynamicDesc.size() - execDynamicParam.outDynamicOffset.back();

  const aclTensorDesc** aclDynamciCompileTensorInputDescArr = 
      new const aclTensorDesc*[inputDynamicNum];
  const aclTensorDesc** aclDynamciCompileTensorOutputDescArr = 
      new const aclTensorDesc*[outputDynamicNum];

  const aclTensorDesc** aclDynamicRunTensorInputDescArr =
      new const aclTensorDesc*[inputDynamicNum];
  const aclTensorDesc** aclDynamicRunTensorOutputDescArr =
      new const aclTensorDesc*[outputDynamicNum];
  const aclDataBuffer** aclDynamicRunDataInputBuffArr =
      new const aclDataBuffer*[inputDynamicNum];
  aclDataBuffer** aclDynamicRunDataOutputBuffArr
      = new aclDataBuffer*[outputDynamicNum];

  std::copy(
      execDynamicParam.inDynamicDesc.begin() + execDynamicParam.inDynamicOffset.back(),
      execDynamicParam.inDynamicDesc.end(),
      aclDynamicRunTensorInputDescArr);

  std::copy(
      execDynamicParam.inDynamicBuffer.begin() + execDynamicParam.inDynamicOffset.back(),
      execDynamicParam.inDynamicBuffer.end(),
      aclDynamicRunDataInputBuffArr);

  std::copy(
      execDynamicParam.outDynamicDesc.begin() + execDynamicParam.outDynamicOffset.back(),
      execDynamicParam.outDynamicDesc.end(),
      aclDynamicRunTensorOutputDescArr);

  std::copy(
      execDynamicParam.outDynamicBuffer.begin() + execDynamicParam.outDynamicOffset.back(),
      execDynamicParam.outDynamicBuffer.end(),
      aclDynamicRunDataOutputBuffArr);

  std::copy(
      execDynamicParam.inDynamicCompileDesc.begin() + execDynamicParam.inDynamicOffset.back(),
      execDynamicParam.inDynamicCompileDesc.end(),
      aclDynamciCompileTensorInputDescArr);

  std::copy(
      execDynamicParam.outDynamicCompileDesc.begin() + execDynamicParam.outDynamicOffset.back(),
      execDynamicParam.outDynamicCompileDesc.end(),
      aclDynamciCompileTensorOutputDescArr);

  params.dynamicParam.input_num = inputDynamicNum;
  params.dynamicParam.output_num = outputDynamicNum;
  params.dynamicParam.input_desc = aclDynamicRunTensorInputDescArr;
  params.dynamicParam.input_data_buf = aclDynamicRunDataInputBuffArr;
  params.dynamicParam.output_desc = aclDynamicRunTensorOutputDescArr;
  params.dynamicParam.output_data_buf = aclDynamicRunDataOutputBuffArr;

  params.dynamicParam.compile_input_desc = aclDynamciCompileTensorInputDescArr;
  params.dynamicParam.compile_output_desc = aclDynamciCompileTensorOutputDescArr;
  params.dynamicParam.dynamicKey = *(dynamicKeys.begin() + execDynamicParam.opCountDynamicOffset.back());
  if (!FuzzyCompileBlacklist::GetInstance().IsInBlacklist(params.opDynamicType) && env::CheckFuzzyEnable()) {
    params.isFuzzy = true;
  }
}
  
void OpDynamicCommandImpl::ReleaseDynamicSource(bool no_blocking) {
  if (no_blocking) {
    std::for_each(
        execDynamicParam.inDynamicCompileDesc.begin() + execDynamicParam.inDynamicOffset.back(), 
        execDynamicParam.inDynamicCompileDesc.end(), 
        aclDestroyTensorDesc);
    std::for_each(
        execDynamicParam.outDynamicCompileDesc.begin() + execDynamicParam.outDynamicOffset.back(),
        execDynamicParam.outDynamicCompileDesc.end(),
        aclDestroyTensorDesc);
    std::for_each(
        execDynamicParam.inDynamicDesc.begin() + execDynamicParam.inDynamicOffset.back(),
        execDynamicParam.inDynamicDesc.end(),
        aclDestroyTensorDesc);
    std::for_each(
        execDynamicParam.outDynamicDesc.begin() + execDynamicParam.outDynamicOffset.back(),
        execDynamicParam.outDynamicDesc.end(),
        aclDestroyTensorDesc);

    if (execDynamicParam.dynamicCompileAttrs.back() != nullptr) {
      aclopDestroyAttr(execDynamicParam.dynamicCompileAttrs.back());
      execDynamicParam.dynamicCompileAttrs.back() = nullptr;
    }

    if (execDynamicParam.dynamicRunAttrs.back() != nullptr) {
      aclopDestroyAttr(execDynamicParam.dynamicRunAttrs.back());
      execDynamicParam.dynamicRunAttrs.back() = nullptr;
    }
  }

  execDynamicParam.inDynamicCompileDesc.erase(
      execDynamicParam.inDynamicCompileDesc.begin() + execDynamicParam.inDynamicOffset.back(), 
      execDynamicParam.inDynamicCompileDesc.end());
  execDynamicParam.inDynamicDesc.erase(
      execDynamicParam.inDynamicDesc.begin() + execDynamicParam.inDynamicOffset.back(), 
      execDynamicParam.inDynamicDesc.end());
  execDynamicParam.inDynamicBuffer.erase(
      execDynamicParam.inDynamicBuffer.begin() + execDynamicParam.inDynamicOffset.back(), 
      execDynamicParam.inDynamicBuffer.end());
  execDynamicParam.inDynamicDims.erase(
      execDynamicParam.inDynamicDims.begin() + execDynamicParam.inDynamicOffset.back(), 
      execDynamicParam.inDynamicDims.end());
  execDynamicParam.outDynamicCompileDesc.erase(
      execDynamicParam.outDynamicCompileDesc.begin() + execDynamicParam.outDynamicOffset.back(), 
      execDynamicParam.outDynamicCompileDesc.end());
  execDynamicParam.outDynamicDesc.erase(
      execDynamicParam.outDynamicDesc.begin() + execDynamicParam.outDynamicOffset.back(), 
      execDynamicParam.outDynamicDesc.end());
  execDynamicParam.outDynamicBuffer.erase(
      execDynamicParam.outDynamicBuffer.begin() + execDynamicParam.outDynamicOffset.back(),
      execDynamicParam.outDynamicBuffer.end());
  execDynamicParam.outDynamicDims.erase(
      execDynamicParam.outDynamicDims.begin() + execDynamicParam.outDynamicOffset.back(), 
      execDynamicParam.outDynamicDims.end());

  opDynamicNames.pop_back();
  dynamicKeys.pop_back();
  execDynamicParam.opCountDynamicOffset.pop_back();
  execDynamicParam.inDynamicOffset.pop_back();
  execDynamicParam.outDynamicOffset.pop_back();
  execDynamicParam.dynamicCompileAttrs.pop_back();
  execDynamicParam.dynamicRunAttrs.pop_back();
}

void OpDynamicCommandImpl::UpdateDynamicParam() {
  execDynamicParam.inDynamicOffset.push_back(
      execDynamicParam.inDynamicOffset.back() + execDynamicParam.inDynamicDims.size());
  execDynamicParam.outDynamicOffset.push_back(
      execDynamicParam.outDynamicOffset.back() + execDynamicParam.outDynamicDims.size());
  execDynamicParam.opCountDynamicOffset.push_back(execDynamicParam.opCountDynamicOffset.back() + dynamicKeys.size());
  dynamicKeys.push_back("");
  execDynamicParam.dynamicCompileAttrs.push_back(nullptr);
  execDynamicParam.dynamicRunAttrs.push_back(nullptr);
}

void OpDynamicCommandImpl::InitDynamicAttr() {
  if (execDynamicParam.dynamicCompileAttrs.back() == nullptr) {
    execDynamicParam.dynamicCompileAttrs.back() = aclopCreateAttr();
  }

  if (execDynamicParam.dynamicRunAttrs.back() == nullptr) {
    execDynamicParam.dynamicRunAttrs.back() = aclopCreateAttr();
  }
}

} // namespace npu
} // namespace native
} // namespace at
