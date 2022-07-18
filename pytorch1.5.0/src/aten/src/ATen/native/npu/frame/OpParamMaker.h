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

#ifndef __NATIVE_NPU_UTILS_OP_PARAM_MAKER__
#define __NATIVE_NPU_UTILS_OP_PARAM_MAKER__

#include <third_party/acl/inc/acl/acl_base.h>
#include "ATen/native/npu/interface/AclOpCompileInterface.h"
#include "ATen/native/npu/interface/EnvVariables.h"
#include "ATen/native/npu/frame/NPUDefine.h"
#include "ATen/native/npu/utils/NpuFuzzyBlacklist.h"
#include "c10/npu/NPUStream.h"
#include "c10/npu/OptionsManager.h"

namespace at {
namespace native {
namespace npu {

// This file is defined wrapper C++ functions of ACL
//
class OpAttrMaker {
 public:
  static void Set(aclopAttr* attr, const string& name, bool value);
  static void Set(aclopAttr* attr, const string& name, int64_t value);
  static void Set(aclopAttr* attr, const string& name, float value);
  static void Set(aclopAttr* attr, const string& name, string& value);
  static void Set(aclopAttr* attr, const string& name, IntArrayRef value);
  static void Set(aclopAttr* attr, const string& name, at::ArrayRef<float> value);
  static void Set(aclopAttr* attr, const string& name, at::ArrayRef<uint8_t> value);
  static void Set(aclopAttr* attr, const string& name, Scalar value);
  static void Set(aclopAttr* attr, const string& name, ScalarType value);
  static void Set(
      aclopAttr* attr,
      const string& name,
      at::ArrayRef<IntArrayRef> value);
}; // class OpAttrMaker

//
class AclTensorDescMaker {
 public:
  AclTensorDescMaker() {}
  ~AclTensorDescMaker() = default;

  AclTensorDescMaker& Create(aclDataType dataType, NPUStorageDesc storageDesc) {
    auto& dims = storageDesc.base_sizes_;
    auto format = storageDesc.origin_format_;
    desc = aclCreateTensorDesc(dataType, dims.size(), dims.data(), format);
    return *this;
  }

  AclTensorDescMaker& Create(
      aclDataType dataType,
      IntArrayRef dims,
      aclFormat format) {
    desc = aclCreateTensorDesc(dataType, dims.size(), dims.data(), format);
    return *this;
  }

  AclTensorDescMaker& Create(aclDataType dataType, aclFormat format) {
    desc = aclCreateTensorDesc(dataType, 0, nullptr, format);
    return *this;
  }

  AclTensorDescMaker SetFormat(aclFormat format) {
    aclSetTensorFormat(desc, format);
    return *this;
  }

  AclTensorDescMaker SetPlacement(aclMemType memType) {
    aclSetTensorPlaceMent(desc, memType);
    return *this;
  }

  template <unsigned int N>
  AclTensorDescMaker& SetShape(const SmallVector<int64_t, N>& dims) {
    aclSetTensorShape(desc, dims.size(), dims.data());
    return *this;
  }

  template <unsigned int N>
  AclTensorDescMaker& SetRange(const SmallVector<int64_t, N>& rangs) {
    int arryDim = rangs.size() == 0 ? 0 : rangs.size() / 2;

    int64_t range[arryDim][2];
    for (int i = 0, j = 0; i < arryDim; i++, j += 2) {
      range[i][0] = rangs[j];
      range[i][1] = rangs[j + 1];
    }

    aclSetTensorShapeRange(desc, arryDim, range);
    return *this;
  }

  AclTensorDescMaker& SetName(const string& name) {
    if (name != "") {
      aclSetTensorDescName(desc, name.c_str());
    }
    return *this;
  }

  AclTensorDescMaker& SetConstAttr(c10::optional<Tensor> cpu_tensor) {
    if (cpu_tensor.has_value() && cpu_tensor.value().defined()) {
      aclSetTensorConst(
          desc,
          cpu_tensor.value().data_ptr(),
          cpu_tensor.value().itemsize() * cpu_tensor.value().numel());
    }

    return *this;
  }

  aclTensorDesc* Get() const {
    return desc;
  }

 private:
  aclTensorDesc* desc = nullptr;

}; // class AclTensorDescMaker

//
class AclTensorBufferMaker {
 public:
  // offset = 0
  explicit AclTensorBufferMaker(const Tensor* tensor, int64_t n = 1) {
    if (tensor == nullptr || n == 0) {
      ptr = aclCreateDataBuffer(nullptr, 0);
    } else {
      ptr = aclCreateDataBuffer(
          (void*)(tensor->data_ptr()), tensor->itemsize() * n);
    }
  }

  // offset = 0
  explicit AclTensorBufferMaker(const Tensor& tensor, int64_t n = 1) {
    ptr =
        aclCreateDataBuffer((void*)(tensor.data_ptr()), tensor.itemsize() * n);
  }

  ~AclTensorBufferMaker() = default;

  aclDataBuffer* Get() const {
    return ptr;
  }

 private:
  aclDataBuffer* ptr = nullptr;

}; // class AclTensorBufferMaker

// the member in AclExecParam is create by :
// aclCreateDataBuffer and aclCreateTensorDesc
// so aclDestroyTensorDesc and aclDestroyDataBuffer should be called when dtr
// aclopDestroyAttr
class OpCommandImpl {
 public:
  OpCommandImpl() {}
  ~OpCommandImpl() {
    // do nothing, can not release resource, because of multi-thread or
    // queue-enable
  }

  void SetName(const string& name) {
    opName = name;
  }

  void AddInput(
      const aclTensorDesc* desc,
      const aclDataBuffer* buffer) {
    execParam.inDesc.emplace_back(std::move(desc));
    execParam.inBuffer.emplace_back(std::move(buffer));
  }

  void AddInput(
      const aclTensorDesc* desc,
      const aclDataBuffer* buffer,
      const Tensor& hostTensor) {
    AddInput(desc, buffer);
    execParam.hostMem.emplace_back(hostTensor);
  }

  void AddOutput(
      const aclTensorDesc* desc,
      aclDataBuffer* buffer) {
    execParam.outDesc.emplace_back(std::move(desc));
    execParam.outBuffer.emplace_back(std::move(buffer));
  }

  template <typename dataType>
  void AddAttr(const string& attrName, dataType value) {
    InitAttr();
    OpAttrMaker::Set(execParam.attr, attrName, value);
  }

  // export op execute params
  void ExportParams(ExecuteParas& params) {
    params.opType = opName;
    params.attr = execParam.attr;

    // make params
    int inputNum = execParam.inDesc.size();
    int outputNum = execParam.outDesc.size();

    size_t inputTensorDescArrLen = inputNum * sizeof(uintptr_t);
    size_t inputDataBuffArrLen   = inputNum * sizeof(uintptr_t);

    size_t outputTensorDescArrLen = outputNum * sizeof(uintptr_t);
    size_t outputDataBuffArrLen   = outputNum * sizeof(uintptr_t);

    size_t totalMemLen = inputTensorDescArrLen + inputDataBuffArrLen +
        outputTensorDescArrLen + outputDataBuffArrLen;
    char* basePtr = static_cast<char* >(malloc(totalMemLen));
    AT_ASSERT(basePtr != nullptr);
    const aclTensorDesc** aclTensorInputDescArr = reinterpret_cast<const aclTensorDesc** >(basePtr);
    basePtr += inputTensorDescArrLen;
    const aclDataBuffer** aclDataInputBuffArr = reinterpret_cast<const aclDataBuffer** >(basePtr);
    basePtr += inputDataBuffArrLen;

    const aclTensorDesc** aclTensorOutputDescArr = reinterpret_cast<const aclTensorDesc** >(basePtr);
    basePtr += outputTensorDescArrLen;
    aclDataBuffer** aclDataOutputBuffArr = reinterpret_cast<aclDataBuffer** >(basePtr);

    std::copy(
        execParam.inDesc.begin(),
        execParam.inDesc.end(),
        aclTensorInputDescArr);
    std::copy(
        execParam.inBuffer.begin(),
        execParam.inBuffer.end(),
        aclDataInputBuffArr);
    std::copy(
        execParam.outDesc.begin(),
        execParam.outDesc.end(),
        aclTensorOutputDescArr);
    std::copy(
        execParam.outBuffer.begin(),
        execParam.outBuffer.end(),
        aclDataOutputBuffArr);

    params.paras.input_num = inputNum;
    params.paras.output_num = outputNum;
    params.paras.input_desc = aclTensorInputDescArr;
    params.paras.input_data_buf = aclDataInputBuffArr;
    params.paras.output_desc = aclTensorOutputDescArr;
    params.paras.output_data_buf = aclDataOutputBuffArr;
    params.hostMemory = execParam.hostMem;
    if (!FuzzyCompileBlacklist::GetInstance().IsInBlacklist(opName) && env::CheckFuzzyEnable()) {
      params.isFuzzy = true;
    }
  }

  void Run(bool sync, SmallVector<int64_t, N> &output_sync_index, SmallVector<Tensor, N> &output_sync_tensor);

  void releaseSource(bool no_blocking = true) {
    if (no_blocking) {
      std::for_each(
          execParam.inDesc.begin(),
          execParam.inDesc.end(),
          aclDestroyTensorDesc);
      std::for_each(
          execParam.outDesc.begin(),
          execParam.outDesc.end(),
          aclDestroyTensorDesc);
      std::for_each(
          execParam.inBuffer.begin(),
          execParam.inBuffer.end(),
          aclDestroyDataBuffer);
      std::for_each(
          execParam.outBuffer.begin(),
          execParam.outBuffer.end(),
          aclDestroyDataBuffer);
      if (execParam.attr != nullptr) {
        aclopDestroyAttr(execParam.attr);
        execParam.attr = nullptr;
      }
    }

    execParam.inDesc.clear();
    execParam.inBuffer.clear();

    execParam.outDesc.clear();
    execParam.outBuffer.clear();
    execParam.hostMem.clear();

    // recover
    execParam.attr = nullptr;
    opName = "";
  }

 private:
  struct AclExecParam {
    SmallVector<const aclTensorDesc*, N> inDesc; // owned
    SmallVector<const aclDataBuffer*, N> inBuffer; // owned
    SmallVector<const aclTensorDesc*, N> outDesc; // owned
    SmallVector<aclDataBuffer*, N> outBuffer; // owned
    SmallVector<Tensor, N> hostMem;
    aclopAttr* attr = nullptr;
  };

  void InitAttr() {
    if (execParam.attr == nullptr) {
      execParam.attr = aclopCreateAttr();
    }
  }

  aclError InnerRun(
    string name, 
    AclExecParam& params, 
    bool sync, 
    SmallVector<int64_t, N> &output_sync_index, 
    SmallVector<Tensor, N> &output_sync_tensor
  );

 private:
  string opName;
  AclExecParam execParam;
}; // class OpCommandImpl


// This class maintain the position of the current
// OpCommandImpl object in vector, the resources in
// the object is
class OpCommandImpls {
public:
  static OpCommandImpls* GetInstance();
  void Push(OpCommandImpl*& ptr);
  void Pop();

private:
  int32_t offset = -1;
  SmallVector<OpCommandImpl, N> objs;
}; // class OpCommandImpls
} // namespace npu
} // namespace native
} // namespace at

#endif