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
#include <Python.h>
#include "c10/npu/NPUQueue.h"
#include "c10/npu/NPUCachingAllocator.h"
#include "c10/npu/interface/AsyncTaskQueueInterface.h"
#include "c10/npu/NPUEventManager.h"
#include "c10/npu/NPUQueue.h"
#include <torch/csrc/autograd/record_function.h>
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/interface/EnvVariables.h"
#include "ATen/native/npu/nputools/AoeUtils.h"
#include "ATen/native/npu/nputools/E2eProfiler.h"
#include "THNPU/THNPUCachingHostAllocator.h"

using namespace c10::npu::queue;

namespace at {
namespace native {
namespace npu {
constexpr size_t MAX_VAL_SIZE = (sizeof(ExecuteParas) > sizeof(CopyParas)) ?
    ((sizeof(ExecuteParas) >  sizeof(EventParas)) ? sizeof(ExecuteParas) : sizeof(EventParas)) :
    ((sizeof(CopyParas) > sizeof(EventParas)) ? sizeof(CopyParas) : sizeof(EventParas));

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

void OpAttrMaker::Set(aclopAttr* attr, const string& name, at::ArrayRef<uint8_t> value) {
  auto vec = value.vec();
  aclopSetAttrListBool(attr, name.c_str(), vec.size(), vec.data());
}

void OpAttrMaker::Set(aclopAttr* attr, const string& name, Scalar value) {
  float val = CalcuOpUtil::get_scalar_float_value(value);
  aclopSetAttrFloat(attr, name.c_str(), val);
}

void OpAttrMaker::Set(aclopAttr* attr, const string& name, ScalarType value) {
  aclDataType val = CalcuOpUtil::convert_to_acl_data_type(value);
  aclopSetAttrDataType(attr, name.c_str(), val);
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

void OpCommandImpl::Run() {
  NPU_LOGD("Op %s Run.", opName.c_str());
  RECORD_HOST_FUNCTION(opName, std::vector<c10::IValue>({}));
  E2E_RECORD_FUNCTION(opName);
  if (PyGILState_Check()) {
    // we need to release GIL for NPU to compile op.
    Py_BEGIN_ALLOW_THREADS
    ACL_REQUIRE_OK_OP(InnerRun(opName, execParam), opName.c_str());
    Py_END_ALLOW_THREADS
  } else {
    ACL_REQUIRE_OK_OP(InnerRun(opName, execParam), opName.c_str());
  }
}

aclError OpCommandImpl::InnerRun(string name, AclExecParam& params) {
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
    if (at::native::npu::aoe::aoe_manager().IsAoeEnabled() &&
        at::native::npu::aoe::aoe_manager().IsInWhiltelist(name)) {

      ret = at::native::npu::AclGenGraphAndDumpForOp(
          name.c_str(),
          inputSize,
          params.inDesc.data(),
          params.inBuffer.data(),
          outputSize,
          params.outDesc.data(),
          params.outBuffer.data(),
          params.attr,
          ACL_ENGINE_SYS,
          at::native::npu::aoe::aoe_manager().GetDumpGraphPath().c_str(),
          nullptr);
      if (ret != ACL_ERROR_NONE) {
        C10_NPU_SHOW_ERR_MSG();
        TORCH_CHECK(false, "In aoe mode, AclGenGraphAndDumpForOp failed!");
      }
    }
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
        nullptr,
        stream);
    ++index;
  } while(NpuUtils::IsOomError(ret, index) && (index < NPU_MAX_OP_EXEC_TRY_NUM));
  if (reset_flag) {
    AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_FUZZ);
  }
  return ret;
}

int ExecFunc(QueueParas* in, aclrtStream stream) {
  auto cur_paras = static_cast<ExecuteParas* >(in->paramVal);
  NPU_LOGD("Op %s Run.", cur_paras->opType.c_str());

  aclError ret;

  bool reset_flag = false;
  if (!cur_paras->isFuzzy) {
    AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_DEFAULT);
    reset_flag = true;
  }

  {
    if (at::native::npu::aoe::aoe_manager().IsAoeEnabled() &&
        at::native::npu::aoe::aoe_manager().IsInWhiltelist(cur_paras->opType)) {
      ret = at::native::npu::AclGenGraphAndDumpForOp(
          (cur_paras->opType).c_str(),
          cur_paras->paras.input_num,
          cur_paras->paras.input_desc,
          cur_paras->paras.input_data_buf,
          cur_paras->paras.output_num,
          cur_paras->paras.output_desc,
          cur_paras->paras.output_data_buf,
          cur_paras->attr,
          ACL_ENGINE_SYS,
          at::native::npu::aoe::aoe_manager().GetDumpGraphPath().c_str(),
          nullptr);
      if (ret != ACL_ERROR_NONE) {
        C10_NPU_SHOW_ERR_MSG();
        TORCH_CHECK(false, "In aoe mode, AclGenGraphAndDumpForOp failed!");
      }
    }

    RECORD_HOST_FUNCTION("aclopCompileAndExecute: " + cur_paras->opType, std::vector<c10::IValue>({}));
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
  }

  if (reset_flag) {
    AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_FUZZ);
  }

  if (ret != ACL_ERROR_NONE) {
    C10_NPU_SHOW_ERR_MSG();
  }

  if (ret != 0) {
    std::cout << "---OpName--- " << cur_paras->opType << std::endl;
  }
  return ret;
}

int MemcopyAsyncFunc(QueueParas* in, aclrtStream stream) {
  auto cur_paras = static_cast<CopyParas* >(in->paramVal);
  RECORD_HOST_FUNCTION("aclrtMemcpyAsync", std::vector<c10::IValue>({}));
  E2E_RECORD_FUNCTION("aclrtMemcpyAsync");
  aclError ret = aclrtMemcpyAsync(cur_paras->dst, cur_paras->dstLen, cur_paras->src,
    cur_paras->srcLen, cur_paras->kind, stream);
  if (ret != ACL_ERROR_NONE) {
    C10_NPU_SHOW_ERR_MSG();
  }
  return ret;
}

int RecordEventFunc(QueueParas* in, aclrtStream stream) {
  auto cur_paras = static_cast<EventParas* >(in->paramVal);
  RECORD_HOST_FUNCTION("aclrtRecordEvent", std::vector<c10::IValue>({}));
  E2E_RECORD_FUNCTION("aclrtRecordEvent");
  aclError ret = aclrtRecordEvent(cur_paras->event, stream);
  if (ret != ACL_ERROR_NONE) {
    C10_NPU_SHOW_ERR_MSG();
  }

  // Temporary modification to avoid problem that
  // event must be recorded before query
  if (cur_paras->eventAllocatorType == HOST_ALLOCATOR_EVENT) {
    THNPUCachingHostAllocator_insertCompleteEvent(cur_paras->event);
  } else if (cur_paras->eventAllocatorType == NPU_ALLOCATOR_EVENT) {
    c10::npu::NPUCachingAllocator::NpuAllocatorInsertRecordedEvent(cur_paras->event);
  }

  return ret;
}

int WaitEventFunc(QueueParas* in, aclrtStream stream) {
  auto cur_paras = static_cast<EventParas* >(in->paramVal);
  RECORD_HOST_FUNCTION("aclrtStreamWaitEvent", std::vector<c10::IValue>({}));
  E2E_RECORD_FUNCTION("aclrtStreamWaitEvent");
  aclError ret = aclrtStreamWaitEvent(stream, cur_paras->event);
  if (ret != ACL_ERROR_NONE) {
    C10_NPU_SHOW_ERR_MSG();
  }
  return ret;
}

int LazyDestroyEventFunc(QueueParas* in, aclrtStream stream) {
  auto cur_paras = static_cast<EventParas* >(in->paramVal);
  RECORD_HOST_FUNCTION("LazyDestroyEvent", std::vector<c10::IValue>({}));
  E2E_RECORD_FUNCTION("LazyDestroyEvent");
  aclError ret = c10::npu::NPUEventManager::GetInstance().LazyDestroy(cur_paras->event);
  if (ret != ACL_ERROR_NONE) {
    C10_NPU_SHOW_ERR_MSG();
  }
  return ret;
}

void CopyFunc(void* dst, void* src, uint32_t queueLen) {
  RECORD_HOST_FUNCTION("Enqueue queue_len: " + to_string(queueLen), std::vector<c10::IValue>({}));
  auto dstPtr = static_cast<QueueParas* >(dst);
  auto srcPtr = static_cast<QueueParas* >(src);
  dstPtr->paramVal = static_cast<uint8_t* >(dst) + sizeof(QueueParas);
  if (dstPtr->paramType == COMPILE_AND_EXECUTE) {
    // string or smallvector of struct is used, deconstructor need be called before memset
    (static_cast<ExecuteParas* >(dstPtr->paramVal))->~ExecuteParas();
  }
  dstPtr->paramStream = srcPtr->paramStream;
  dstPtr->paramType = srcPtr->paramType;
  dstPtr->paramLen = srcPtr->paramLen;
  memset(dstPtr->paramVal, 0, MAX_VAL_SIZE);
  if (srcPtr->paramType == COMPILE_AND_EXECUTE) {
    (static_cast<ExecuteParas* >(dstPtr->paramVal))->Copy(*(static_cast<ExecuteParas* >(srcPtr->paramVal)));
  } else if (srcPtr->paramType == ASYNC_MEMCPY) {
    (static_cast<CopyParas* >(dstPtr->paramVal))->Copy(*(static_cast<CopyParas* >(srcPtr->paramVal)));
  } else {
    (static_cast<EventParas* >(dstPtr->paramVal))->Copy(*(static_cast<EventParas* >(srcPtr->paramVal)));
  }
}

void ReleaseFunc(void* ptr, c10::npu::ReleaseQueue& releaseQueue) {
  releaseQueue.PushToReleaseQueue(ptr);
}

void* NewFunc(int caption, int& size) {
  size = sizeof(QueueParas) + MAX_VAL_SIZE;
  void *ptr = malloc(size * caption);
  TORCH_CHECK(ptr != nullptr, "OpCommand new buffer must be not NULL");
  memset(ptr, 0, size * caption);
  return ptr;
}

void DeleteFunc(void* ptr) {
  free(ptr);
}

using Func = int (*)(QueueParas*, aclrtStream);
using AsyncFuncMap = std::map<QueueParamType, Func>;
AsyncFuncMap funcMap = {
  {COMPILE_AND_EXECUTE, ExecFunc},
  {ASYNC_MEMCPY, MemcopyAsyncFunc},
  {RECORD_EVENT, RecordEventFunc},
  {WAIT_EVENT, WaitEventFunc},
  {LAZY_DESTROY_EVENT, LazyDestroyEventFunc},
};

int AsncExecFunc(void* data, uint32_t queueLen) {
  RECORD_HOST_FUNCTION("Dequeue queue_len: " + to_string(queueLen), std::vector<c10::IValue>({}));
  auto queueParam = static_cast<QueueParas* >(data);
  auto type = queueParam->paramType;
  aclrtStream stream = queueParam->paramStream;
  auto ret = funcMap[type](queueParam, stream);
  return ret;
}

void CopyReleaseParamFunc(void* dst, void* src)
{
  auto dstPtr = static_cast<QueueParas* >(dst);
  auto srcPtr = static_cast<QueueParas* >(src);
  dstPtr->paramType = srcPtr->paramType;
  dstPtr->paramVal = static_cast<uint8_t* >(dst) + sizeof(QueueParas);
  if (srcPtr->paramType == COMPILE_AND_EXECUTE) {
    (static_cast<ExecuteParas* >(dstPtr->paramVal))->CopyEx(*(static_cast<ExecuteParas* >(srcPtr->paramVal)));
    (static_cast<ExecuteParas* >(srcPtr->paramVal))->hostMemory.clear();
  }
}

void  ReleaseParamFunc(void* ptr) {
  auto queueParam = static_cast<QueueParas* >(ptr);
  auto type = queueParam->paramType;
  if (type == COMPILE_AND_EXECUTE) {
    auto cur_paras = static_cast<ExecuteParas* >(queueParam->paramVal);
    cur_paras->Release();
  }
}

REGISTER_QUEUE_FUNC(AsncExecFunc, CopyFunc, ReleaseFunc, NewFunc, DeleteFunc,
  CopyReleaseParamFunc, ReleaseParamFunc)

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
} // namespace npu
} // namespace native
} // namespace at