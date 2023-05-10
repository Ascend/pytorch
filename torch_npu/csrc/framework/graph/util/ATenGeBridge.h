// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <c10/core/TensorOptions.h>
#include <torch_npu/csrc/framework/graph/util/NPUGraph.h>
#include <third_party/acl/inc/graph/operator.h>
#include "torch_npu/csrc/core/NPUStorageImpl.h"

namespace at_npu {
namespace native {

class ATenGeBridge {
public:
  static ge::DataType GetGeDType(c10::ScalarType type);

  static ge::DataType GetGeDType(caffe2::TypeMeta type_meta);

  static ge::DataType GetGeDType(const std::string& real_dtype);

  static ge::Shape GetGeShape(c10::ArrayRef<int64_t> vec);

  static ge::TensorDesc InferGeTenosrDesc(
      const torch_npu::NPUStorageDesc& storage_desc,
      const c10::optional<std::string>& real_dtype,
      bool is_op_desc = false);

  static void CheckAndBuildGeOpForNode(NodePtr node,
                                       std::vector<ge::Operator>& const_input_ops);

  static ge::Tensor MakeGeTensor(const ge::TensorDesc& tensor_desc,
                                 const void* device_ptr, const size_t nbytes);
private:
  template <typename T>
  static T TryToGetAnyValue(const Any& any_val) {
    T val;
    try {
      val = CastAs<T>(any_val);
    } catch (AnyCastException& bd) {
      AT_ERROR(bd.what(), typeid(T).name());
    }
    return val;
  }

  template <typename ConstType>
  static ge::Operator SetAndReturnGeOpConstInput(
      const Any& const_input,
      ge::OperatorPtr ge_op);

  static void SetSensitiveFormat(
      const Any& sensitive_format,
      ge::OperatorPtr ge_op,
      NodeExtInfoType ext_type);

  static void PorcessDynamicInputReg(
      NodePtr node,
      ge::OperatorPtr& ge_op,
      std::string op_name);

  template <typename AttrType>
  static void SetGeOpAttr(const Any& attr_val, ge::OperatorPtr ge_op) {
    AttrType attr = TryToGetAnyValue<AttrType>(attr_val);
    ge_op->SetAttr(attr.first.c_str(), attr.second);
  }

  static void AddNodeExtInfoIntoGeOp(
      c10::ArrayRef<std::pair<NodeExtInfoType, Any>> ext_info,
      ge::OperatorPtr ge_op,
      std::vector<ge::Operator>& const_input_ops);
};
} // namespace native
} // namespace at_npu
