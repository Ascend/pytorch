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
  static void Set(aclopAttr* attr, const string& name, Scalar value);
  static void Set(
      aclopAttr* attr,
      const string& name,
      at::ArrayRef<IntArrayRef> value);
}; // class OpAttrMaker

class AttrInfoMaker {
 public:
  static void Add(bool value, string& attrInfo);
  static void Add(int64_t value, string& attrInfo);
  static void Add(float value, string& attrInfo);
  static void Add(string value, string& attrInfo);
  static void Add(IntArrayRef value, string& attrInfo);
  static void Add(at::ArrayRef<float> value,string& attrInfo);
  static void Add(Scalar value, string& attrInfo);
  static void Add(at::ArrayRef<IntArrayRef> value, string& attrInfo);
};

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
  AclTensorBufferMaker(const Tensor* tensor, int64_t n = 1) {
    if (tensor == nullptr || n == 0) {
      ptr = aclCreateDataBuffer(nullptr, 0);
    } else {
      ptr = aclCreateDataBuffer(
          (void*)(tensor->data_ptr()), tensor->itemsize() * n);
    }
  }

  // offset = 0
  AclTensorBufferMaker(const Tensor& tensor, int64_t n = 1) {
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
      const aclDataBuffer* buffer,
      int64_t dim,
      aclFormat format) {
    inputCounter += 1;
    execParam.inDesc.emplace_back(std::move(desc));
    execParam.inBuffer.emplace_back(std::move(buffer));
    execParam.inDims.emplace_back(dim);
    execParam.inFormats.emplace_back(format);
  }

  void AddInput(
      const aclTensorDesc* desc,
      const aclDataBuffer* buffer,
      int64_t dim,
      aclFormat format,
      const Tensor& hostTensor) {
    AddInput(desc, buffer, dim, format);
    execParam.hostMem.emplace_back(hostTensor.storage());
  }

  void AddConst(SmallVector<int64_t, N> dimList) {
    int64_t dimNum = dimList.size();
    int64_t* constList = new int64_t[dimNum];
    for (int i = 0; i < dimNum; ++i) {
      constList[i] = dimList[i];
    }

    execParam.constIdxs.emplace_back(inputCounter);
    execParam.constLists.emplace_back(constList);
  }

  void AddOutput(
      const aclTensorDesc* desc,
      aclDataBuffer* buffer,
      int64_t dim,
      aclFormat format) {
    execParam.outDesc.emplace_back(std::move(desc));
    execParam.outBuffer.emplace_back(std::move(buffer));
    execParam.outDims.emplace_back(dim);
    execParam.outFormats.emplace_back(format);
  }

  template <typename dataType>
  void AddAttr(const string& attrName, dataType value) {
    InitAttr();
    AttrInfoMaker::Add(value, attrInfo);
    OpAttrMaker::Set(execParam.attr, attrName, value);
    execParam.hasAttr = true;
  }

  // export op execute params
  void ExportParams(ExecuteParas& params) {
    InitAttr();
    params.opType = opName;
    params.attrInfo = attrInfo;
    params.attr = execParam.attr;

    // make params
    int inputNum = execParam.inDesc.size();
    int outputNum = execParam.outDesc.size();
    int constNum = execParam.constLists.size();
    const int64_t** constListArr = new const int64_t*[constNum];
    const aclTensorDesc** aclTensorInputDescArr =
        new const aclTensorDesc*[inputNum];
    const aclTensorDesc** aclTensorOutputDescArr =
        new const aclTensorDesc*[outputNum];
    const aclDataBuffer** aclDataInputBuffArr =
        new const aclDataBuffer*[inputNum];
    aclDataBuffer** aclDataOutputBuffArr = new aclDataBuffer*[outputNum];

    int64_t* constIdxArr = new int64_t[constNum];
    int64_t* inputDimsArr = new int64_t[inputNum];
    int64_t* outputDimsArr = new int64_t[outputNum];
    aclFormat* inputFormatsArr = new aclFormat[inputNum];
    aclFormat* outputFormatsArr = new aclFormat[outputNum];

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

    std::copy(
        execParam.inDims.begin(),
        execParam.inDims.end(),
        inputDimsArr);
    std::copy(
        execParam.outDims.begin(),
        execParam.outDims.end(),
        outputDimsArr);
    std::copy(
        execParam.inFormats.begin(),
        execParam.inFormats.end(),
        inputFormatsArr);
    std::copy(
        execParam.outFormats.begin(),
        execParam.outFormats.end(),
        outputFormatsArr);

    std::copy(
        execParam.constLists.begin(),
        execParam.constLists.end(),
        constListArr);
    std::copy(
        execParam.constIdxs.begin(),
        execParam.constIdxs.end(),
        constIdxArr);

    params.paras.input_num = inputNum;
    params.paras.output_num = outputNum;
    params.paras.input_desc = aclTensorInputDescArr;
    params.paras.input_data_buf = aclDataInputBuffArr;
    params.paras.output_desc = aclTensorOutputDescArr;
    params.paras.output_data_buf = aclDataOutputBuffArr;

    params.paras.inputDims = inputDimsArr;
    params.paras.outputDims = outputDimsArr;
    params.paras.inputFormats = inputFormatsArr;
    params.paras.outputFormats = outputFormatsArr;
    params.paras.hasAttr = execParam.hasAttr;

    params.constParams.constNum = constNum;
    params.constParams.constList = constListArr;
    params.constParams.constIdx = constIdxArr;
    params.hostMemory = execParam.hostMem;
    if (!FuzzyCompileBlacklist::GetInstance().IsInBlacklist(opName) && env::CheckFuzzyEnable()) {
      params.isFuzzy = true;
    }
  }

  void Run();

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
      std::for_each(
          execParam.constLists.begin(),
          execParam.constLists.end(),
          [](const int64_t* constList) { delete[] constList; });
      if (execParam.attr != nullptr) {
        aclopDestroyAttr(execParam.attr);
        execParam.attr = nullptr;
      }
    }

    execParam.inDesc.clear();
    execParam.inBuffer.clear();
    execParam.inDims.clear();
    execParam.inFormats.clear();

    execParam.outDesc.clear();
    execParam.outBuffer.clear();
    execParam.outDims.clear();
    execParam.outFormats.clear();

    execParam.constIdxs.clear();
    execParam.constLists.clear();
    execParam.hostMem.clear();

    // recover
    execParam.hasAttr = false;
    execParam.attr = nullptr;
    opName = "";
    attrInfo = "attrs:";
    inputCounter = 0;
  }

 private:
  struct AclExecParam {
    SmallVector<const aclTensorDesc*, N> inDesc; // owned
    SmallVector<const aclDataBuffer*, N> inBuffer; // owned
    SmallVector<const aclTensorDesc*, N> outDesc; // owned
    SmallVector<aclDataBuffer*, N> outBuffer; // owned
    SmallVector<int64_t, N> inDims;
    SmallVector<int64_t, N> outDims;
    SmallVector<aclFormat, N> inFormats;
    SmallVector<aclFormat, N> outFormats;
    SmallVector<const int64_t*, N> constLists;
    SmallVector<int64_t, N> constIdxs;
    SmallVector<Storage, N> hostMem;
    aclopAttr* attr = nullptr;
    bool hasAttr = false;
  };

  void InitAttr() {
    if (execParam.attr == nullptr) {
      execParam.attr = aclopCreateAttr();
    }
  }

  aclError InnerRun(string name, AclExecParam& params);

 private:
  int64_t inputCounter = 0;
  string opName;
  string attrInfo = "attrs:";
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