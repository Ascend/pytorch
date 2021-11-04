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

#include "OpParamMaker.h"
#include <c10/npu/OptionsManager.h>
#include "c10/npu/NPUQueue.h"
#include "c10/npu/NPUCachingAllocator.h"
#include <torch/csrc/autograd/record_function.h>
#include "ATen/native/npu/aoe/AutoTune.h"
#include "ATen/native/npu/utils/DynamicShapeUtil.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/interface/EnvVariables.h"
#include "aten/src/THNPU/THNPUCachingHostAllocator.h"

namespace at {
namespace native {
namespace npu {
class AsyncCopyTask {
public:
  AsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind);
  AsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind, Tensor& holdTensor);
  ~AsyncCopyTask() = default;
  void LaunchCopyTask();
  void LaunchCopyTask(bool isPinnedMem);
private:
  CopyParas copyParam;
};

void OpAttrMaker::Set(aclopAttr* attr, const string& name, bool value) {
  aclopSetAttrBool(attr, name.c_str(), value);
}

void OpAttrMaker::Set(aclopAttr* attr, const string& name, int64_t value) {
  aclopSetAttrInt(attr, name.c_str(), value);
}

void OpAttrMaker::Set(aclopAttr* attr, const string& name, float value) {
  aclopSetAttrFloat(attr, name.c_str(), value);
}

void OpAttrMaker::Set(aclopAttr* attr, const string& name, string& value) {
  aclopSetAttrString(attr, name.c_str(), value.c_str());
}

void OpAttrMaker::Set(aclopAttr* attr, const string& name, IntArrayRef value) {
  auto vec = value.vec();
  aclopSetAttrListInt(attr, name.c_str(), vec.size(), vec.data());
}

void OpAttrMaker::Set(aclopAttr* attr, const string& name, at::ArrayRef<float> value) {
  auto vec = value.vec();
  aclopSetAttrListFloat(attr, name.c_str(), vec.size(), vec.data());
}

void OpAttrMaker::Set(aclopAttr* attr, const string& name, Scalar value) {
  float val = CalcuOpUtil::get_scalar_float_value(value);
  aclopSetAttrFloat(attr, name.c_str(), val);
}


void OpAttrMaker::Set(
    aclopAttr* attr,
    const string& name,
    at::ArrayRef<IntArrayRef> value) {
  // Pointer to values of each listInt.
  SmallVector<int64_t*, N> attrValue;
  // Pointer to number of each listInt.
  SmallVector<int, N> eachListIntNum;
  // Value of each listInt.
  SmallVector<SmallVector<int64_t, N>, N> eachListIntVal;
  for (int i = 0; i < value.size(); i++) {
    SmallVector<int64_t, N> listInt;
    int64_t valueSize = value[i].size();
    listInt.resize(valueSize);
    std::copy(value[i].begin(), value[i].end(), listInt.begin());
    eachListIntVal.emplace_back(listInt);
    attrValue.emplace_back(eachListIntVal.back().data());
    eachListIntNum.emplace_back(valueSize);
  }

  aclopSetAttrListListInt(
        attr,
        name.c_str(),
        attrValue.size(),
        eachListIntNum.data(),
        attrValue.data());
}


void AttrInfoMaker::Add(bool value, string& attrInfo) {
  attrInfo += to_string(value) + "-";
}

void AttrInfoMaker::Add(int64_t value, string& attrInfo) {
  attrInfo += to_string(value) + "-";
}

void AttrInfoMaker::Add(float value, string& attrInfo) {
  attrInfo += to_string(value) + "-";
}

void AttrInfoMaker::Add(string value, string& attrInfo) {
  attrInfo += value + "-";
}

void AttrInfoMaker::Add(IntArrayRef value, string& attrInfo) {
  auto vec = value.vec();
  for (unsigned i = 0; i < vec.size(); i++)
    attrInfo += to_string(vec.at(i)) + ",";
  attrInfo += "-";
}

void AttrInfoMaker::Add(
    at::ArrayRef<float> value,
    string& attrInfo) {
  auto vec = value.vec();
  for (unsigned i = 0; i < vec.size(); i++)
    attrInfo += to_string(vec.at(i)) + ",";
  attrInfo += "-";
}

void AttrInfoMaker::Add(Scalar value, string& attrInfo) {
  float val = CalcuOpUtil::get_scalar_float_value(value);
  attrInfo += to_string(val) + "-";
}

void AttrInfoMaker::Add(
    at::ArrayRef<IntArrayRef> value,
    string& attrInfo) {
  // Pointer to values of each listInt.
  SmallVector<int64_t*, N> attrValue;
  // Pointer to number of each listInt.
  SmallVector<int, N> eachListIntNum;
  // Value of each listInt.
  SmallVector<SmallVector<int64_t, N>, N> eachListIntVal;
  for (int i = 0; i < value.size(); i++) {
    int64_t valueSize = value[i].size();
    attrInfo += to_string(valueSize) + ",";
  }
  attrInfo += "-";
}


void OpCommandImpl::Run() {
  InitAttr();
  NPU_LOGD("Op %s Run.", opName.c_str());
  RECORD_FUNCTION(opName, std::vector<c10::IValue>({}));

  ACL_REQUIRE_OK_OP(InnerRun(opName, execParam), opName.c_str());
}

aclError OpCommandImpl::InnerRun(string name, AclExecParam& params) {
  AutotuneManager::GetInstance()->PushGraph(name, params.graph);
  auto stream = c10::npu::getCurrentNPUStream();
  auto inputSize = params.inBuffer.size();
  auto outputSize = params.outBuffer.size();
  bool reset_flag = false;
  if (FuzzyCompileBlacklist::GetInstance().IsInBlacklist(name) && env::CheckFuzzyEnable()) {
    AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_DEFAULT);
    reset_flag = true;
  }
  aclError ret;
  int index = 0;
  do {
    ret = aclopCompileAndExecute(
      name.c_str(),
      inputSize,
      params.inDesc.data(),
      params.inBuffer.data(),
      outputSize,
      params.outDesc.data(),
      params.outBuffer.data(),
      params.attr,
      ACL_ENGINE_SYS,
      ACL_COMPILE_SYS,
      NULL,
      stream);
    ++index;
  } while(NpuUtils::IsOomError(ret, index) && (index < NPU_MAX_OP_EXEC_TRY_NUM));
  if (reset_flag) {
    AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_FUZZ);
  }
  return ret;
}

int ExecFunc(void* in, aclrtStream stream) {
  auto cur_paras = (ExecuteParas*)(((QueueParas*)in)->paramVal);
  NPU_LOGD("Op %s Run.", cur_paras->opType.c_str());

  aclError ret;
  if (!c10::npu::OptionsManager::CheckDynamicEnable()) {
    bool reset_flag = false;
    if (!cur_paras->isFuzzy) {
      AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_DEFAULT);
      reset_flag = true;
    }
    int index = 0;
    do {
      ret = aclopCompileAndExecute(
        (cur_paras->opType).c_str(),
        cur_paras->paras.input_num,
        cur_paras->paras.input_desc,
        cur_paras->paras.input_data_buf,
        cur_paras->paras.output_num,
        cur_paras->paras.output_desc,
        cur_paras->paras.output_data_buf,
        cur_paras->attr,
        ACL_ENGINE_SYS,
        ACL_COMPILE_SYS,
        nullptr,
        stream);
      ++index;
    } while(NpuUtils::IsOomError(ret, index) && (index < NPU_MAX_OP_EXEC_TRY_NUM));
    if (reset_flag) {
      AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_FUZZ);
    }
    if (ret != ACL_ERROR_NONE) {
      C10_NPU_SHOW_ERR_MSG();
    }
  } else {
    ret = DynamicRun(*cur_paras, stream);
  }

  if (ret != 0) {
    std::cout << "---OpName--- " << cur_paras->opType << std::endl;
  }
  return ret;
}

int MemcopyAsyncFunc(void* in, aclrtStream stream) {
  auto cur_paras = (CopyParas*)(((QueueParas*)in)->paramVal);
  aclError ret = aclrtMemcpyAsync(cur_paras->dst, cur_paras->dstLen, cur_paras->src,
    cur_paras->srcLen, cur_paras->kind, stream);
  if (ret != ACL_ERROR_NONE) {
    C10_NPU_SHOW_ERR_MSG();
  }
  return ret;
}

int RecordEventFunc(void* in, aclrtStream stream) {
  auto cur_paras = (EventParas*)(((QueueParas*)in)->paramVal);
  aclError ret = aclrtRecordEvent(cur_paras->event, stream);
  if (ret != ACL_ERROR_NONE) {
    C10_NPU_SHOW_ERR_MSG();
  }
  return ret;
}

void CopyFunc(void* dst, void* src, uint32_t queueLen) {
  RECORD_FUNCTION("Enqueue queue_len: " + to_string(queueLen), std::vector<c10::IValue>({}));
  auto dstPtr = (QueueParas*)dst;
  auto srcPtr = (QueueParas*)src;
  dstPtr->paramVal = ((int8_t*)dst) + sizeof(QueueParas);
  if (dstPtr->paramType == COMPILE_AND_EXECUTE) {
    ((ExecuteParas*)(dstPtr->paramVal))->hostMemory.clear();
  }
  if (dstPtr->paramType == ASYNC_MEMCPY_EX) {
    ((CopyParas*)(dstPtr->paramVal))->pinMem.clear();
  }
  dstPtr->paramType = srcPtr->paramType;
  dstPtr->paramLen = srcPtr->paramLen;
  memset(dstPtr->paramVal, 0, sizeof(ExecuteParas));
  if (srcPtr->paramType == COMPILE_AND_EXECUTE) {
    ((ExecuteParas*)dstPtr->paramVal)->Copy(*((ExecuteParas*)srcPtr->paramVal));
  } else if (srcPtr->paramType == ASYNC_MEMCPY) {
    memcpy(dstPtr->paramVal, srcPtr->paramVal, sizeof(CopyParas));
  } else if (srcPtr->paramType == ASYNC_MEMCPY_EX) {
    ((CopyParas*)dstPtr->paramVal)->Copy(*((CopyParas*)srcPtr->paramVal));
  }else {
    memcpy(dstPtr->paramVal, srcPtr->paramVal, sizeof(EventParas));
  }
}

void ReleaseFunc(void* ptr) {
  auto type = ((QueueParas *)ptr)->paramType;
  if (type == COMPILE_AND_EXECUTE) {
    auto cur_paras = (ExecuteParas*)(((QueueParas *)ptr)->paramVal);
    if (!cur_paras->opDynamicType.empty()) {
      cur_paras->DynamicRelease();
      cur_paras->opDynamicType = "";
    }
    cur_paras->Release();
  }
}

void* NewFunc(int caption, int& size) {
  size_t maxSize = sizeof(ExecuteParas) > sizeof(CopyParas) ? sizeof(ExecuteParas) : sizeof(CopyParas);
  maxSize = maxSize > sizeof(EventParas) ? maxSize : sizeof(EventParas);
  size = sizeof(QueueParas) + maxSize;
  void *ptr = malloc(size * caption);
  memset(ptr, 0, size * caption);
  return ptr;
}

void DeleteFunc(void* ptr) {
  free(ptr);
}

typedef int (*Func)(void*, aclrtStream);
using AsyncFuncMap = std::map<QueueParamType, Func>;
AsyncFuncMap funcMap = {
  {COMPILE_AND_EXECUTE, ExecFunc},
  {ASYNC_MEMCPY, MemcopyAsyncFunc},
  {ASYNC_MEMCPY_EX, MemcopyAsyncFunc},
  {RECORD_EVENT, RecordEventFunc}
};

int AsncExecFunc(void* data, aclrtStream stream) {
  auto type = ((QueueParas *)data)->paramType;
  auto ret = funcMap[type](data, stream);
  return ret;
}

REGISTER_QUEUE_FUNC(AsncExecFunc, CopyFunc, ReleaseFunc, NewFunc, DeleteFunc)

OpCommandImpls* OpCommandImpls::GetInstance() {
  static OpCommandImpls impl;
  return &impl;
}

void OpCommandImpls::Push(OpCommandImpl*& ptr) {
  offset += 1;
  if (objs.size() <= offset) {
    OpCommandImpl impl;
    objs.emplace_back(impl);
  }
  TORCH_CHECK(
      objs.size() > offset,
      "OpCommand size (",
      objs.size(),
      ") is smaller than offset (",
      offset,
      ")");
  ptr = &objs[offset];
}

void OpCommandImpls::Pop() {
  TORCH_CHECK(
      offset >= 0, "OpCommand current offset should not be less than ", offset);
  offset -= 1;
}

AsyncCopyTask::AsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind)
{
  copyParam.dst = dst;
  copyParam.dstLen = dstLen;
  copyParam.src = src;
  copyParam.srcLen = srcLen;
  copyParam.kind = kind;
}

AsyncCopyTask::AsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind,
    Tensor& holdTensor)
{
  copyParam.dst = dst;
  copyParam.dstLen = dstLen;
  copyParam.src = src;
  copyParam.srcLen = srcLen;
  copyParam.kind = kind;
  copyParam.pinMem.emplace_back(holdTensor);
}

void AsyncCopyTask::LaunchCopyTask()
{
  if (c10::npu::OptionsManager::CheckQueueEnable()) {
    QueueParas params(ASYNC_MEMCPY, sizeof(CopyParas), &copyParam);
    c10::npu::enCurrentNPUStream(&params);
  } else {
    c10::npu::NPUStream stream = c10::npu::getCurrentNPUStream();
    AT_NPU_CHECK(aclrtMemcpyAsync(
      copyParam.dst,
      copyParam.dstLen,
      copyParam.src,
      copyParam.srcLen,
      copyParam.kind,
      stream));
  }
}

void AsyncCopyTask::LaunchCopyTask(bool isPinnedMem)
{
  if (c10::npu::OptionsManager::CheckQueueEnable() && isPinnedMem) {
    QueueParas params(ASYNC_MEMCPY_EX, sizeof(CopyParas), &copyParam);
    c10::npu::enCurrentNPUStream(&params);
    copyParam.pinMem.clear();
  } else {
    c10::npu::NPUStream stream = c10::npu::getCurrentNPUStream();
    AT_NPU_CHECK(aclrtMemcpyAsync(
      copyParam.dst,
      copyParam.dstLen,
      copyParam.src,
      copyParam.srcLen,
      copyParam.kind,
      stream));
  }
}

aclError LaunchAsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind)
{
  npu::AsyncCopyTask copyTask(dst, dstLen, src, srcLen, kind);
  copyTask.LaunchCopyTask();
  return ACL_ERROR_NONE;
}

aclError LaunchAsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind,
    Tensor& holdTensor, bool isPinMem)
{
  AsyncCopyTask copyTask(dst, dstLen, src, srcLen, kind, holdTensor);
  copyTask.LaunchCopyTask(isPinMem);
  return ACL_ERROR_NONE;
}
} // namespace npu
} // namespace native
} // namespace at