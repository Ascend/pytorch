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

#ifndef __NATIVE_NPU_UTILS_OP_DYNAMIC_PARAM_MAKER__
#define __NATIVE_NPU_UTILS_OP_DYNAMIC_PARAM_MAKER__

#include <third_party/acl/inc/acl/acl_base.h>
#include <third_party/acl/inc/acl/acl_op_compiler.h>
#include "ATen/native/npu/frame/OpParamMaker.h"
#include "c10/npu/NPUStream.h"

namespace at {
namespace native {
namespace npu {

// the member in AclExecParam is create by :
// aclCreateDataBuffer and aclCreateTensorDesc
// so aclDestroyTensorDesc and aclDestroyDataBuffer should be called when dtr
// aclopDestroyAttr
class OpDynamicCommandImpl {
 public:
  static OpDynamicCommandImpl* GetInstance() {
    static OpDynamicCommandImpl impl;
    return &impl;
  }

  ~OpDynamicCommandImpl() {
    // do nothing, can not release resource, because of multi-thread or
    // queue-enable
  }

  void SetDynamicName(string& name);
  const string SetDynamicName();

  template <typename dataType>
  void AddDynamicAttr(string attrName, dataType value) {
    InitDynamicAttr();
    AttrInfoMaker::Add(value, dynamicKeys.back());
    OpAttrMaker::Set(
        execDynamicParam.dynamicCompileAttrs.back(), attrName, value);
    OpAttrMaker::Set(execDynamicParam.dynamicRunAttrs.back(), attrName, value);
  }

  void AddDynamicInput(
      const aclTensorDesc* desc,
      const aclDataBuffer* buffer,
      int64_t dim,
      aclFormat format);

  void AddDynamicCompileInputDesc(const aclTensorDesc* desc);

  void AddDynamicOutputDesc(const aclTensorDesc* desc);

  void AddDynamicKey(string dynamicKey);

  void AddDynamicOutput(
      const aclTensorDesc* desc,
      aclDataBuffer* buffer,
      int64_t dim,
      aclFormat format);

  // export op execute params
  void ExportDynamicParams(ExecuteParas& params);

  void ReleaseDynamicSource(bool no_blocking = true);

  void UpdateDynamicParam();

  const string& GetDynamicName();

 private:
  OpDynamicCommandImpl() {}
  struct AclDynamicExecParam {
    SmallVector<const aclTensorDesc*, N> inDynamicDesc; // owned
    SmallVector<const aclDataBuffer*, N> inDynamicBuffer; // owned
    SmallVector<const aclTensorDesc*, N> outDynamicDesc; // owned
    SmallVector<aclDataBuffer*, N> outDynamicBuffer; // owned
    SmallVector<aclopAttr*, N> dynamicCompileAttrs;
    SmallVector<aclopAttr*, N> dynamicRunAttrs;
    SmallVector<int64_t, N> inDynamicDims;
    SmallVector<int64_t, N> outDynamicDims;
    SmallVector<aclFormat, N> inDynamicFormats;
    SmallVector<aclFormat, N> outDynamicFormats;

    SmallVector<uint32_t, N> inDynamicOffset{0};
    SmallVector<uint32_t, N> outDynamicOffset{0};
    SmallVector<uint32_t, N> opCountDynamicOffset{0};

    SmallVector<const aclTensorDesc*, N> inDynamicCompileDesc; // owned
    SmallVector<const aclTensorDesc*, N> outDynamicCompileDesc; // owned
  };

  void InitDynamicAttr();

 private:
  SmallVector<string, N> opDynamicNames;
  SmallVector<string, N> dynamicKeys = {};
  AclDynamicExecParam execDynamicParam;
}; // class OpCommandImpl

} // namespace npu
} // namespace native
} // namespace at

#endif