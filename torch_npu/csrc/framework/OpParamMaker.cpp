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

#include <c10/npu/NPUQueue.h>
#include <ATen/record_function.h>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/aoe/AoeUtils.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/OpParamMaker.h"
#include "torch_npu/csrc/core/npu/THNPUCachingHostAllocator.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "c10/npu/NPUEventManager.h"
#include "c10/npu/interface/AsyncTaskQueueInterface.h"

namespace at_npu
{
  namespace native
  {

    void OpAttrMaker::Set(aclopAttr *attr, const string &name, bool value)
    {
      aclopSetAttrBool(attr, name.c_str(), value);
    }

    void OpAttrMaker::Set(aclopAttr *attr, const string &name, int64_t value)
    {
      aclopSetAttrInt(attr, name.c_str(), value);
    }

    void OpAttrMaker::Set(aclopAttr *attr, const string &name, float value)
    {
      aclopSetAttrFloat(attr, name.c_str(), value);
    }

    void OpAttrMaker::Set(aclopAttr *attr, const string &name, string value)
    {
      aclopSetAttrString(attr, name.c_str(), value.c_str());
    }

    void OpAttrMaker::Set(aclopAttr *attr, const string &name, c10::IntArrayRef value)
    {
      auto vec = value.vec();
      aclopSetAttrListInt(attr, name.c_str(), vec.size(), vec.data());
    }

    void OpAttrMaker::Set(aclopAttr *attr, const string &name, at::ArrayRef<float> value)
    {
      auto vec = value.vec();
      aclopSetAttrListFloat(attr, name.c_str(), vec.size(), vec.data());
    }

    void OpAttrMaker::Set(aclopAttr *attr, const string &name, c10::Scalar value)
    {
      float val = CalcuOpUtil::get_scalar_float_value(value);
      aclopSetAttrFloat(attr, name.c_str(), val);
    }

    void OpAttrMaker::Set(
        aclopAttr *attr,
        const string &name,
        at::ArrayRef<c10::IntArrayRef> value)
    {
      // Pointer to values of each listInt.
      c10::SmallVector<int64_t *, N> attrValue;
      // Pointer to number of each listInt.
      c10::SmallVector<int, N> eachListIntNum;
      // Value of each listInt.
      c10::SmallVector<c10::SmallVector<int64_t, N>, N> eachListIntVal;
      for (int i = 0; i < value.size(); i++)
      {
        c10::SmallVector<int64_t, N> listInt;
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

    void AttrInfoMaker::Add(bool value, string &attrInfo)
    {
      attrInfo += std::to_string(value) + "-";
    }

    void AttrInfoMaker::Add(int64_t value, string &attrInfo)
    {
      attrInfo += std::to_string(value) + "-";
    }

    void AttrInfoMaker::Add(float value, string &attrInfo)
    {
      attrInfo += std::to_string(value) + "-";
    }

    void AttrInfoMaker::Add(string value, string &attrInfo)
    {
      attrInfo += value + "-";
    }

    void AttrInfoMaker::Add(c10::IntArrayRef value, string &attrInfo)
    {
      auto vec = value.vec();
      for (unsigned i = 0; i < vec.size(); i++)
        attrInfo += std::to_string(vec.at(i)) + ",";
      attrInfo += "-";
    }

    void AttrInfoMaker::Add(
        at::ArrayRef<float> value,
        string &attrInfo)
    {
      auto vec = value.vec();
      for (unsigned i = 0; i < vec.size(); i++)
        attrInfo += std::to_string(vec.at(i)) + ",";
      attrInfo += "-";
    }

    void AttrInfoMaker::Add(c10::Scalar value, string &attrInfo)
    {
      float val = CalcuOpUtil::get_scalar_float_value(value);
      attrInfo += std::to_string(val) + "-";
    }

    void AttrInfoMaker::Add(
        at::ArrayRef<c10::IntArrayRef> value,
        string &attrInfo)
    {
      // Pointer to values of each listInt.
      c10::SmallVector<int64_t *, N> attrValue;
      // Pointer to number of each listInt.
      c10::SmallVector<int, N> eachListIntNum;
      // Value of each listInt.
      c10::SmallVector<c10::SmallVector<int64_t, N>, N> eachListIntVal;
      for (int i = 0; i < value.size(); i++)
      {
        int64_t valueSize = value[i].size();
        attrInfo += std::to_string(valueSize) + ",";
      }
      attrInfo += "-";
    }

    void OpCommandImpl::Run()
    {
      InitAttr();
      NPU_LOGD("Op %s Run.", opName.c_str());
      RECORD_FUNCTION(opName, std::vector<c10::IValue>({}));

      ACL_REQUIRE_OK_OP(InnerRun(opName, execParam), opName.c_str());
    }

    aclError OpCommandImpl::InnerRun(string name, AclExecParam &params)
    {
      auto stream = c10::npu::getCurrentNPUStream();
      auto inputSize = params.inBuffer.size();
      auto outputSize = params.outBuffer.size();
      bool reset_flag = false;
      if (FuzzyCompileBlacklist::GetInstance().IsInBlacklist(name) && env::CheckFuzzyEnable())
      {
        AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_DEFAULT);
        reset_flag = true;
      }
      aclError ret;
      int index = 0;
      do
      {
        if (at_npu::native::aoe::aoe_manager().IsAoeEnabled() &&
            !at_npu::native::aoe::aoe_manager().IsInBlacklist(name)) {
          ret = at_npu::native::AclGenGraphAndDumpForOp(
              name.c_str(),
              inputSize,
              params.inDesc.data(),
              params.inBuffer.data(),
              outputSize,
              params.outDesc.data(),
              params.outBuffer.data(),
              params.attr,
              ACL_ENGINE_SYS,
              at_npu::native::aoe::aoe_manager().GetDumpGraphPath().c_str(),
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
            NULL,
            stream);
        ++index;
      } while (NpuUtils::IsOomError(ret, index) && (index < NPU_MAX_OP_EXEC_TRY_NUM));
      if (reset_flag)
      {
        AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_FUZZ);
      }
      return ret;
    }

    int ExecFunc(c10::npu::queue::QueueParas* in, aclrtStream stream)
    {
      auto cur_paras = static_cast<ExecuteParas* >(in->paramVal);
      NPU_LOGD("Op %s Run.", cur_paras->opType.c_str());

      aclError ret;
      bool reset_flag = false;
      if (!cur_paras->isFuzzy)
      {
        AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_DEFAULT);
        reset_flag = true;
      }
      if (at_npu::native::aoe::aoe_manager().IsAoeEnabled() &&
          !at_npu::native::aoe::aoe_manager().IsInBlacklist(cur_paras->opType)) {
        ret = at_npu::native::AclGenGraphAndDumpForOp(
            (cur_paras->opType).c_str(),
            cur_paras->paras.input_num,
            cur_paras->paras.input_desc,
            cur_paras->paras.input_data_buf,
            cur_paras->paras.output_num,
            cur_paras->paras.output_desc,
            cur_paras->paras.output_data_buf,
            cur_paras->attr,
            ACL_ENGINE_SYS,
            at_npu::native::aoe::aoe_manager().GetDumpGraphPath().c_str(),
            nullptr);
        if (ret != ACL_ERROR_NONE) {
          C10_NPU_SHOW_ERR_MSG();
          TORCH_CHECK(false, "In aoe mode, AclGenGraphAndDumpForOp failed!");
        }
      }
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
      if (reset_flag)
      {
        AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_FUZZ);
      }

      if (ret != ACL_ERROR_NONE)
      {
        C10_NPU_SHOW_ERR_MSG();
      }

      if (ret != 0)
      {
        std::cout << "---OpName--- " << cur_paras->opType << std::endl;
      }
      return ret;
    }

    int MemcopyAsyncFunc(c10::npu::queue::QueueParas* in, aclrtStream stream)
    {
      auto cur_paras = static_cast<c10::npu::queue::CopyParas* >(in->paramVal);
      aclError ret = aclrtMemcpyAsync(cur_paras->dst, cur_paras->dstLen, cur_paras->src,
        cur_paras->srcLen, cur_paras->kind, stream);
      if (ret != ACL_ERROR_NONE) {
        C10_NPU_SHOW_ERR_MSG();
      }
      return ret;
    }

    int RecordEventFunc(c10::npu::queue::QueueParas* in, aclrtStream stream)
    {
      auto cur_paras = static_cast<c10::npu::queue::EventParas* >(in->paramVal);
      aclError ret = aclrtRecordEvent(cur_paras->event, stream);
      if (ret != ACL_ERROR_NONE) {
        C10_NPU_SHOW_ERR_MSG();
      }

      // Temporary modification to avoid problem that
      // event must be recorded before query
      if (cur_paras->eventAllocatorType == c10::npu::queue::HOST_ALLOCATOR_EVENT) {
        THNPUCachingHostAllocator_insertCompleteEvent(cur_paras->event);
      } else if (cur_paras->eventAllocatorType == c10::npu::queue::NPU_ALLOCATOR_EVENT) {
        c10_npu::NPUCachingAllocator::NpuAllocatorInsertRecordedEvent(cur_paras->event);
      }

      return ret;
    }

    int WaitEventFunc(c10::npu::queue::QueueParas* in, aclrtStream stream) {
      auto cur_paras = static_cast<c10::npu::queue::EventParas* >(in->paramVal);
      aclError ret = aclrtStreamWaitEvent(stream, cur_paras->event);
      if (ret != ACL_ERROR_NONE) {
        C10_NPU_SHOW_ERR_MSG();
      }
      return ret;
    }

    int LazyDestroyEventFunc(c10::npu::queue::QueueParas* in, aclrtStream stream) {
      auto cur_paras = static_cast<c10::npu::queue::EventParas* >(in->paramVal);
      aclError ret = c10::npu::NPUEventManager::GetInstance().LazyDestroy(cur_paras->event);
      if (ret != ACL_ERROR_NONE) {
        C10_NPU_SHOW_ERR_MSG();
      }
      return ret;
    }

    size_t GetMaxLen(size_t x, size_t y, size_t z)
    {
      return x > y ? (x > z ? x : z) : (y > z ? y : z);
    }

    void CopyFunc(void* dst, void* src, c10::SmallVector<c10::Storage, N>& needClearVec, uint32_t queueLen)
    {
      auto dstPtr = static_cast<c10::npu::queue::QueueParas* >(dst);
      auto srcPtr = static_cast<c10::npu::queue::QueueParas* >(src);
      dstPtr->paramVal = static_cast<uint8_t* >(dst) + sizeof(c10::npu::queue::QueueParas);
      // pin memory free will add aclrtRecordEvent to queue
      // in order to avoid deadlock, pin memory free operation is moved out of the enqueue operation
      if (dstPtr->paramType == c10::npu::queue::COMPILE_AND_EXECUTE) {
        needClearVec.swap((static_cast<ExecuteParas* >(dstPtr->paramVal))->hostMemory);
        // string or smallvector of struct is used, deconstructor need be called before memset
        (static_cast<ExecuteParas* >(dstPtr->paramVal))->~ExecuteParas();
      } else if (dstPtr->paramType == c10::npu::queue::ASYNC_MEMCPY_EX) {
        needClearVec.swap((static_cast<c10::npu::queue::CopyParas* >(dstPtr->paramVal))->pinMem);
        // string or smallvector of struct is used, deconstructor need be called before memset
        (static_cast<c10::npu::queue::CopyParas* >(dstPtr->paramVal))->~CopyParas();
      }
      dstPtr->paramStream = srcPtr->paramStream;
      dstPtr->paramType = srcPtr->paramType;
      dstPtr->paramLen = srcPtr->paramLen;
      size_t maxSize = GetMaxLen(sizeof(ExecuteParas), sizeof(c10::npu::queue::CopyParas),
          sizeof(c10::npu::queue::EventParas));
      memset(dstPtr->paramVal, 0, maxSize);
      if (srcPtr->paramType == c10::npu::queue::COMPILE_AND_EXECUTE) {
        (static_cast<ExecuteParas* >(dstPtr->paramVal))->Copy(*(static_cast<ExecuteParas* >(srcPtr->paramVal)));
      } else if ((srcPtr->paramType == c10::npu::queue::ASYNC_MEMCPY) ||
        (srcPtr->paramType == c10::npu::queue::ASYNC_MEMCPY_EX)) {
        (static_cast<c10::npu::queue::CopyParas* >(dstPtr->paramVal))->
            Copy(*(static_cast<c10::npu::queue::CopyParas* >(srcPtr->paramVal)));
      } else {
        (static_cast<c10::npu::queue::EventParas* >(dstPtr->paramVal))->
            Copy(*(static_cast<c10::npu::queue::EventParas* >(srcPtr->paramVal)));
      }
    }

    void ReleaseFunc(void* ptr, c10::npu::ReleaseQueue& releaseQueue)
    {
      releaseQueue.PushToReleaseQueue(ptr);
    }

    void* NewFunc(int caption, int& size)
    {
      size_t maxSize = GetMaxLen(sizeof(ExecuteParas), sizeof(c10::npu::queue::CopyParas),
          sizeof(c10::npu::queue::EventParas));
      size = sizeof(c10::npu::queue::QueueParas) + maxSize;
      void *ptr = malloc(size * caption);
      TORCH_CHECK(ptr != nullptr, "OpCommand new buffer must be not NULL");
      memset(ptr, 0, size * caption);
      return ptr;
    }

    void DeleteFunc(void* ptr)
    {
      free(ptr);
    }

    typedef int (*Func)(c10::npu::queue::QueueParas*, aclrtStream);
    using AsyncFuncMap = std::map<c10::npu::queue::QueueParamType, Func>;
    AsyncFuncMap funcMap = {
      {c10::npu::queue::COMPILE_AND_EXECUTE, ExecFunc},
      {c10::npu::queue::ASYNC_MEMCPY, MemcopyAsyncFunc},
      {c10::npu::queue::ASYNC_MEMCPY_EX, MemcopyAsyncFunc},
      {c10::npu::queue::RECORD_EVENT, RecordEventFunc},
      {c10::npu::queue::WAIT_EVENT, WaitEventFunc},
      {c10::npu::queue::LAZY_DESTROY_EVENT, LazyDestroyEventFunc},
    };

    int AsncExecFunc(void* data, uint32_t queueLen) {
      auto queueParam = static_cast<c10::npu::queue::QueueParas* >(data);
      auto type = queueParam->paramType;
      aclrtStream stream = queueParam->paramStream;
      auto ret = funcMap[type](queueParam, stream);
      return ret;
    }

    void CopyReleaseParamFunc(void* dst, void* src)
    {
      auto dstPtr = static_cast<c10::npu::queue::QueueParas* >(dst);
      auto srcPtr = static_cast<c10::npu::queue::QueueParas* >(src);
      dstPtr->paramType = srcPtr->paramType;
      dstPtr->paramVal = static_cast<uint8_t* >(dst) + sizeof(c10::npu::queue::QueueParas);
      if (srcPtr->paramType == c10::npu::queue::COMPILE_AND_EXECUTE) {
        (static_cast<ExecuteParas* >(dstPtr->paramVal))->CopyEx(*(static_cast<ExecuteParas* >(srcPtr->paramVal)));
      }
    }

    void  ReleaseParamFunc(void* ptr) {
      auto queueParam = static_cast<c10::npu::queue::QueueParas* >(ptr);
      auto type = queueParam->paramType;
      if (type == c10::npu::queue::COMPILE_AND_EXECUTE) {
        auto cur_paras = static_cast<ExecuteParas* >(queueParam->paramVal);
        cur_paras->Release();
      }
    }

    REGISTER_QUEUE_FUNC(AsncExecFunc, CopyFunc, ReleaseFunc, NewFunc, DeleteFunc,
      CopyReleaseParamFunc, ReleaseParamFunc)

    OpCommandImpls *OpCommandImpls::GetInstance()
    {
      static OpCommandImpls impl;
      return &impl;
    }

    void OpCommandImpls::Push(OpCommandImpl *&ptr)
    {
      offset += 1;
      if (objs.size() <= offset)
      {
        OpCommandImpl impl;
        objs.push_back(impl);
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

    void OpCommandImpls::Pop()
    {
      TORCH_CHECK(
          offset >= 0, "OpCommand current offset should not be less than ", offset);
      offset -= 1;
    }

  } // namespace native
} // namespace at_npu