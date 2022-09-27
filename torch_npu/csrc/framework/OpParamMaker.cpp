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

#include "torch_npu/csrc/core/npu/NPUQueue.h"
#include <ATen/record_function.h>
#include <Python.h>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/aoe/AoeUtils.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/OpParamMaker.h"
#include "torch_npu/csrc/core/npu/THNPUCachingHostAllocator.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"

namespace at_npu
{
  namespace native
  {
    using namespace c10_npu::queue;
    constexpr size_t MAX_VAL_SIZE = (sizeof(ExecuteParas) > sizeof(CopyParas)) ?
      ((sizeof(ExecuteParas) >  sizeof(EventParas)) ? sizeof(ExecuteParas) : sizeof(EventParas)) :
      ((sizeof(CopyParas) > sizeof(EventParas)) ? sizeof(CopyParas) : sizeof(EventParas));

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
      aclopSetAttrListInt(attr, name.c_str(), value.size(), value.data());
    }

    void OpAttrMaker::Set(aclopAttr *attr, const string &name, at::ArrayRef<float> value)
    {
      aclopSetAttrListFloat(attr, name.c_str(), value.size(), value.data());
    }

    void OpAttrMaker::Set(aclopAttr* attr, const string& name, at::ArrayRef<uint8_t> value)
    {
      aclopSetAttrListBool(attr, name.c_str(), value.size(), value.data());
    }

    void OpAttrMaker::Set(aclopAttr *attr, const string &name, c10::Scalar value)
    {
      float val = CalcuOpUtil::get_scalar_float_value(value);
      aclopSetAttrFloat(attr, name.c_str(), val);
    }

    void OpAttrMaker::Set(aclopAttr* attr, const string& name, at::ScalarType value) 
    {
      aclDataType val = CalcuOpUtil::convert_to_acl_data_type(value);
      aclopSetAttrDataType(attr, name.c_str(), val);
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

    void OpCommandImpl::Run(
        bool sync, 
        c10::SmallVector<int64_t, N> &sync_index, 
        c10::SmallVector<at::Tensor, N> &outputTensor) {
      NPU_LOGD("Op %s Run.", opName.c_str());
      RECORD_FUNCTION(opName, std::vector<c10::IValue>({}));
      if (PyGILState_Check()) {
        // we need to release GIL for NPU to compile op.
        Py_BEGIN_ALLOW_THREADS
        ACL_REQUIRE_OK_OP(InnerRun(opName, execParam, sync, sync_index, outputTensor), opName.c_str());
        Py_END_ALLOW_THREADS
      } else {
        ACL_REQUIRE_OK_OP(InnerRun(opName, execParam, sync, sync_index, outputTensor), opName.c_str());
      }
    }

    aclError OpCommandImpl::InnerRun(
        string name, 
        AclExecParam &params, 
        bool sync, 
        c10::SmallVector<int64_t, N> &sync_index, 
        c10::SmallVector<at::Tensor, N> &outputTensor) {
      auto stream = c10_npu::getCurrentNPUStream();
      if (stream.isDataPreprocessStream()) {
        OpAttrMaker::Set(params.attr, "_performance_prior", "true");
        OpAttrMaker::Set(params.attr, "_exclude_engines", "AICORE");
      }
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
            at_npu::native::aoe::aoe_manager().IsInWhiltelist(name)) {
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
        if (!sync) {
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
        } else {
          int64_t dimSize;
          ret = AclopCompileAndExecuteV2(
              name.c_str(),
              inputSize,
              const_cast<aclTensorDesc**>(params.inDesc.data()),
              const_cast<aclDataBuffer**>(params.inBuffer.data()),
              outputSize,
              const_cast<aclTensorDesc**>(params.outDesc.data()),
              params.outBuffer.data(),
              params.attr,
              ACL_ENGINE_SYS,
              ACL_COMPILE_SYS,
              NULL,
              stream);

          for (size_t i = 0; i < sync_index.size(); i++) {
            c10::SmallVector<int64_t, N> real_shape;
            for (int64_t j = 0; j < outputTensor[sync_index[i]].dim(); j++) {
              C10_NPU_CHECK(aclGetTensorDescDimV2(params.outDesc[sync_index[i]], j, &dimSize));
              real_shape.emplace_back(dimSize);
            }
            outputTensor[sync_index[i]].resize_(real_shape);
          }
        }
        ++index;
      } while (NpuUtils::IsOomError(ret, index) && (index < NPU_MAX_OP_EXEC_TRY_NUM));
      if (reset_flag)
      {
        AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_FUZZ);
      }
      return ret;
    }

    int ExecFunc(c10_npu::queue::QueueParas* in, aclrtStream stream)
    {
      auto cur_paras = static_cast<ExecuteParas* >(in->paramVal);
      NPU_LOGD("Op %s Run.", cur_paras->opType.c_str());

      if (cur_paras->isDataPreprocessOp) {
        OpAttrMaker::Set(const_cast<aclopAttr*>(cur_paras->attr), "_performance_prior", "true");
        OpAttrMaker::Set(const_cast<aclopAttr*>(cur_paras->attr), "_exclude_engines", "AICORE");
      }
      aclError ret;
      bool reset_flag = false;
      if (!cur_paras->isFuzzy)
      {
        AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_DEFAULT);
        reset_flag = true;
      }
      if (at_npu::native::aoe::aoe_manager().IsAoeEnabled() &&
          at_npu::native::aoe::aoe_manager().IsInWhiltelist(cur_paras->opType)) {
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

    int MemcopyAsyncFunc(c10_npu::queue::QueueParas* in, aclrtStream stream)
    {
      auto cur_paras = static_cast<c10_npu::queue::CopyParas* >(in->paramVal);
      aclError ret = aclrtMemcpyAsync(cur_paras->dst, cur_paras->dstLen, cur_paras->src,
        cur_paras->srcLen, cur_paras->kind, stream);
      if (ret != ACL_ERROR_NONE) {
        C10_NPU_SHOW_ERR_MSG();
      }
      return ret;
    }

    int RecordEventFunc(c10_npu::queue::QueueParas* in, aclrtStream stream)
    {
      auto cur_paras = static_cast<c10_npu::queue::EventParas* >(in->paramVal);
      aclError ret = aclrtRecordEvent(cur_paras->event, stream);
      if (ret != ACL_ERROR_NONE) {
        C10_NPU_SHOW_ERR_MSG();
      }

      // Temporary modification to avoid problem that
      // event must be recorded before query
      if (cur_paras->eventAllocatorType == c10_npu::queue::HOST_ALLOCATOR_EVENT) {
        THNPUCachingHostAllocator_insertCompleteEvent(cur_paras->event);
      } else if (cur_paras->eventAllocatorType == c10_npu::queue::NPU_ALLOCATOR_EVENT) {
        c10_npu::NPUCachingAllocator::NpuAllocatorInsertRecordedEvent(cur_paras->event);
      }

      return ret;
    }

    int WaitEventFunc(c10_npu::queue::QueueParas* in, aclrtStream stream) {
      auto cur_paras = static_cast<c10_npu::queue::EventParas* >(in->paramVal);
      aclError ret = aclrtStreamWaitEvent(stream, cur_paras->event);
      if (ret != ACL_ERROR_NONE) {
        C10_NPU_SHOW_ERR_MSG();
      }
      return ret;
    }

    int LazyDestroyEventFunc(c10_npu::queue::QueueParas* in, aclrtStream stream) {
      auto cur_paras = static_cast<c10_npu::queue::EventParas* >(in->paramVal);
      aclError ret = c10_npu::NPUEventManager::GetInstance().LazyDestroy(cur_paras->event);
      if (ret != ACL_ERROR_NONE) {
        C10_NPU_SHOW_ERR_MSG();
      }
      return ret;
    }

    void CopyFunc(void* dst, void* src, uint32_t queueLen)
    {
      auto dstPtr = static_cast<c10_npu::queue::QueueParas* >(dst);
      auto srcPtr = static_cast<c10_npu::queue::QueueParas* >(src);
      dstPtr->paramVal = static_cast<uint8_t* >(dst) + sizeof(c10_npu::queue::QueueParas);
      dstPtr->paramStream = srcPtr->paramStream;
      dstPtr->paramType = srcPtr->paramType;
      dstPtr->paramLen = srcPtr->paramLen;
      if (srcPtr->paramType == c10_npu::queue::COMPILE_AND_EXECUTE) {
        new(dstPtr->paramVal) ExecuteParas();
        (static_cast<ExecuteParas* >(dstPtr->paramVal))->Copy(*(static_cast<ExecuteParas* >(srcPtr->paramVal)));
      } else if ((srcPtr->paramType == c10_npu::queue::ASYNC_MEMCPY)) {
        new(dstPtr->paramVal) CopyParas();
        (static_cast<c10_npu::queue::CopyParas* >(dstPtr->paramVal))->
            Copy(*(static_cast<c10_npu::queue::CopyParas* >(srcPtr->paramVal)));
      } else {
        new(dstPtr->paramVal) EventParas();
        (static_cast<c10_npu::queue::EventParas* >(dstPtr->paramVal))->
            Copy(*(static_cast<c10_npu::queue::EventParas* >(srcPtr->paramVal)));
      }
    }

    void ReleaseFunc(void* ptr, c10_npu::ReleaseQueue& releaseQueue)
    {
      releaseQueue.PushToReleaseQueue(ptr);
    }

    void* NewFunc(int caption, int& size)
    {
      size = sizeof(c10_npu::queue::QueueParas) + MAX_VAL_SIZE;
      void *ptr = malloc(size * caption);
      TORCH_CHECK(ptr != nullptr, "OpCommand new buffer must be not NULL");
      memset(ptr, 0, size * caption);
      return ptr;
    }

    void DeleteFunc(void* ptr)
    {
      free(ptr);
    }

    using Func = int(*)(c10_npu::queue::QueueParas*, aclrtStream);
    using AsyncFuncMap = std::map<c10_npu::queue::QueueParamType, Func>;
    AsyncFuncMap funcMap = {
      {c10_npu::queue::COMPILE_AND_EXECUTE, ExecFunc},
      {c10_npu::queue::ASYNC_MEMCPY, MemcopyAsyncFunc},
      {c10_npu::queue::RECORD_EVENT, RecordEventFunc},
      {c10_npu::queue::WAIT_EVENT, WaitEventFunc},
      {c10_npu::queue::LAZY_DESTROY_EVENT, LazyDestroyEventFunc},
    };

    int AsncExecFunc(void* data, uint32_t queueLen) {
      auto queueParam = static_cast<c10_npu::queue::QueueParas* >(data);
      auto type = queueParam->paramType;
      aclrtStream stream = queueParam->paramStream;
      auto ret = funcMap[type](queueParam, stream);
      return ret;
    }

    void CopyReleaseParamFunc(void* dst, void* src)
    {
      auto dstPtr = static_cast<c10_npu::queue::QueueParas* >(dst);
      auto srcPtr = static_cast<c10_npu::queue::QueueParas* >(src);
      dstPtr->paramType = srcPtr->paramType;
      dstPtr->paramVal = static_cast<uint8_t* >(dst) + sizeof(c10_npu::queue::QueueParas);
      if (srcPtr->paramType == c10_npu::queue::COMPILE_AND_EXECUTE) {
        (static_cast<ExecuteParas* >(dstPtr->paramVal))->CopyEx(*(static_cast<ExecuteParas* >(srcPtr->paramVal)));
        (static_cast<ExecuteParas* >(srcPtr->paramVal))->hostMemory.clear();
      }
    }

    void  ReleaseParamFunc(void* ptr) {
      auto queueParam = static_cast<c10_npu::queue::QueueParas* >(ptr);
      auto type = queueParam->paramType;
      if (type == c10_npu::queue::COMPILE_AND_EXECUTE) {
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
      ++offset;
      if (objs.size() <= offset)
      {
        OpCommandImpl impl;
        objs.emplace_back(std::move(impl));
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