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

#include <ATen/native/npu/utils/CalcuOpUtil.h>
#include <ATen/native/npu/utils/NpuUtils.h>

#include <ATen/ATen.h>
#include <c10/npu/NPUGraph.h>
namespace at {
namespace native {
namespace npu {
using c10::npu::graph::DynamicInputRegFunc;
using c10::npu::graph::DyNumAndIndex;
using c10::npu::graph::NodeExtInfoType;
using c10::npu::graph::NodePtr;

class OperatorAttrMaker {
public:
  static void SetAttr(const string& attr_name, bool value, NodePtr node) {
    node->AddExtInfo(
        NodeExtInfoType::ATTR_TYPE_BOOL, std::make_pair(attr_name, value));
    node->UpdateNodeHash(value);
  }

  static void SetAttr(const string& attr_name, float value, NodePtr node) {
    node->AddExtInfo(
        NodeExtInfoType::ATTR_TYPE_FLOAT, std::make_pair(attr_name, value));
    node->UpdateNodeHash(value);
  }

  static void SetAttr(const string& attr_name, int64_t value, NodePtr node) {
    node->AddExtInfo(
        NodeExtInfoType::ATTR_TYPE_LONG, std::make_pair(attr_name, value));
    node->UpdateNodeHash(value);
  }

  static void SetAttr(
      const string& attr_name,
      const string& value,
      NodePtr node) {
    node->AddExtInfo(
        NodeExtInfoType::ATTR_TYPE_STRING, std::make_pair(attr_name, value));
    node->UpdateNodeHash(value);
  }

  static void SetAttr(
      const string& attr_name,
      const ArrayRef<int64_t>& value,
      NodePtr node) {
    vector<int64_t> val(value.begin(), value.end());
    node->AddExtInfo(
        NodeExtInfoType::ATTR_TYPE_LIST_LONG,
        std::make_pair(attr_name, std::move(val)));
    node->UpdateNodeHash(val);
  }

  static void SetAttr(
      const string& attr_name,
      const ArrayRef<float>& value,
      NodePtr node) {
    vector<float> val(value.begin(), value.end());
    node->AddExtInfo(
        NodeExtInfoType::ATTR_TYPE_LIST_FLOAT,
        std::make_pair(attr_name, std::move(val)));
    node->UpdateNodeHash(val);
  }

  static void SetAttr(
      const string& attr_name,
      const ArrayRef<uint8_t>& value,
      NodePtr node) {
    vector<bool> val;
    val.reserve(value.size());
    for (auto item : value) {
      val.push_back(item != 0);
    }
    node->AddExtInfo(
        NodeExtInfoType::ATTR_TYPE_LIST_BOOL,
        std::make_pair(attr_name, std::move(val)));
    node->UpdateNodeHash(value);
  }
  
  static void SetAttr(
      const string& attr_name,
      const Scalar& value,
      NodePtr node) {
    float val = CalcuOpUtil::get_scalar_float_value(value);
    node->AddExtInfo(
        NodeExtInfoType::ATTR_TYPE_FLOAT, std::make_pair(attr_name, val));
    node->UpdateNodeHash(val);
  }
};

class GraphCommandImpl {
public:
  GraphCommandImpl() = default;
  ~GraphCommandImpl() = default;

  void SetName(const std::string& name);

  void AddInput();

  void AddInput(
      const at::Tensor& input,
      const string& desc_name,
      const string& real_dtype,
      const optional<aclFormat>& sensitive_format = nullopt);

  void AddInput(
      const Scalar& input,
      const ScalarType type,
      CompileType compile_type);

  void AddInput(const IntArrayRef& dim_list, const ScalarType to_type);

  void AddOutput(
      const at::Tensor& output,
      const string& desc_name = "",
      const string& real_type = "",
      const optional<aclFormat>& sensitive_format = nullopt);

  void AddDynamicInputRegFunc(
      DynamicInputRegFunc func,
      DyNumAndIndex num_and_index);

  void ReduceScalarValue(
      const Scalar& input,
      const ScalarType type,
      uint32_t& host_ptr_offset);

  template <typename T>
  void AddAttr(const string& attr_name, T value) {
    OperatorAttrMaker::SetAttr(attr_name, value, ir_node_);
  }

private:
  void AddZeroDimInput(const at::Tensor& input, const string& desc_name);

  uint32_t output_index_ = 0;
  uint32_t input_index_ = 0;
  NodePtr ir_node_ = nullptr;
};
} // namespace npu
} // namespace native
} // namespace at