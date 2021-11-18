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
#include <c10/npu/NPUGraph.h>
#include <third_party/acl/inc/graph/operator.h>

namespace at {
namespace native {
namespace npu {

using c10::npu::graph::NodeExtInfoType;
using c10::npu::graph::DyNumAndIndex;
using c10::npu::graph::DynamicInputRegFunc;
using c10::npu::graph::NodePtr;

class ATenGeBridge {
public:
  static ge::DataType GetGeDType(ScalarType type);

  static ge::DataType GetGeDType(caffe2::TypeMeta type_meta);

  static ge::DataType GetGeDType(const string& real_dtype);

  static ge::Shape GetGeShape(ArrayRef<int64_t> vec);

  static ge::TensorDesc InferGeTenosrDesc(
      const NPUStorageDesc& storage_desc,
      const caffe2::TypeMeta& type_meta,
      const c10::optional<string>& real_dtype);

  static void CheckAndBuildGeOpForNode(NodePtr node);

private:
  template <typename T>
  static T TryToGetAnyValue(const c10::any& any_val) {
    T val;
    try {
      val = any_cast<T>(any_val);
    } catch (bad_any_cast &bd) {
      AT_ERROR(bd.what(), typeid(T).name());
    }
    return val;
  }

  template <typename ConstType>
  static void SetGeOpConstInput(
      const c10::any& const_input,
      ge::OperatorPtr ge_op);

  static void SetSensitiveFormat(
      const c10::any& sensitive_format,
      ge::OperatorPtr ge_op,
      NodeExtInfoType ext_type);

  static void PorcessDynamicInputReg(
      NodePtr node,
      ge::OperatorPtr& ge_op,
      string op_name);

  template <typename AttrType>
  static void SetGeOpAttr(const c10::any& attr_val, ge::OperatorPtr ge_op) {
    AttrType attr = TryToGetAnyValue<AttrType>(attr_val);
    ge_op->SetAttr(attr.first.c_str(), attr.second);
  }

  static void AddNodeExtInfoIntoGeOp(
      ArrayRef<std::pair<NodeExtInfoType, c10::any>> ext_info,
      ge::OperatorPtr ge_op);
};
} // namespace npu
} // namespace native
} // namespace at
