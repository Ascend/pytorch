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

#ifndef __NATIVE_NPU_UTILS_OP_COMMAND_HELPER__
#define __NATIVE_NPU_UTILS_OP_COMMAND_HELPER__

#include "ATen/native/npu/utils/NpuUtils.h"
#include <ATen/ATen.h>
#include <third_party/acl/inc/acl/acl.h>
#include <third_party/acl/inc/acl/acl_base.h>

namespace at {
namespace native {
namespace npu {

// covert pytorch tensor to acl tensor.
class OpCmdHelper {
public:
  static std::tuple<aclTensorDesc*, aclDataBuffer*>
  CovertTensorToAclInput(
      const Tensor& tensor,
      const c10::optional<Tensor>& cpu_tensor,
      const string& descName,
      const string& forceDataType = "");

  static std::tuple<aclTensorDesc*, aclDataBuffer*>
  CovertTensorWithZeroDimToAclInput(const Tensor& tensor, ScalarType type);

  static std::tuple<aclTensorDesc*, aclDataBuffer*>
  CovertNPUTensorWithZeroDimToAclInput(const Tensor& tensor, const string& descName);

  static std::tuple<aclTensorDesc*, aclDataBuffer*>
  CovertScalarToAclInput(const Tensor& tensor, ScalarType type);

  static std::tuple<aclTensorDesc*, aclDataBuffer*>
  CovertToAclOutput(const Tensor& tensor, const string& forceDataType);

  static std::tuple<aclTensorDesc*, aclDataBuffer*>
  CovertTransDataTensorToAcl(
      const Tensor& tensor);

  static std::tuple<aclTensorDesc*, aclDataBuffer*>
  CovertHostTensorToAclInput(
      const Tensor& tensor, ScalarType type, CompileType compileType, const string& forceDataType);
}; // class OpCommandImpl

} // npu
} // native
} // at

#endif