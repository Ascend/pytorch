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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/graph/util/NPUGraph.h"
#include "torch_npu/csrc/framework/graph/util/ATenGeBridge.h"
#include "torch_npu/csrc/framework/graph/scalar/ScalarMemoryOps.h"
#include <ATen/ATen.h>

namespace at_npu {
namespace native {

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
      const c10::ArrayRef<int64_t>& value,
      NodePtr node) {
    vector<int64_t> val(value.begin(), value.end());
    node->AddExtInfo(
        NodeExtInfoType::ATTR_TYPE_LIST_LONG,
        std::make_pair(attr_name, std::move(val)));
    node->UpdateNodeHash(val);
  }

  static void SetAttr(
      const string& attr_name,
      const c10::ArrayRef<float>& value,
      NodePtr node) {
    vector<float> val(value.begin(), value.end());
    node->AddExtInfo(
        NodeExtInfoType::ATTR_TYPE_LIST_FLOAT,
        std::make_pair(attr_name, std::move(val)));
    node->UpdateNodeHash(val);
  }

  static void SetAttr(
      const string& attr_name,
      const c10::ArrayRef<uint8_t>& value,
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
      const c10::Scalar& value,
      NodePtr node) {
    float val = CalcuOpUtil::GetScalarFloatValue(value);
    node->AddExtInfo(
        NodeExtInfoType::ATTR_TYPE_FLOAT, std::make_pair(attr_name, val));
    node->UpdateNodeHash(val);
  }

  static void SetAttr(
    const string& attr_name,
    const c10::ScalarType& value,
    NodePtr node) {
    ge::DataType val = ATenGeBridge::GetGeDType(value);
    node->AddExtInfo(
        NodeExtInfoType::ATTR_TYPE_DATATYPE, std::make_pair(attr_name, val));
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
      const c10::optional<aclFormat>& sensitive_format = c10::nullopt);

  void AddInput(
      const c10::Scalar& input,
      const at::ScalarType type,
      CompileType compile_type);

  void AddInput(const c10::IntArrayRef& dim_list, const at::ScalarType to_type);

  void AddInput(const string& str);

  void AddOutput(
      const at::Tensor& output,
      const string& desc_name = "",
      const string& real_type = "",
      const c10::optional<aclFormat>& sensitive_format = c10::nullopt);

  void AddDynamicInputRegFunc(
      DynamicInputRegFunc func,
      DyNumAndIndex num_and_index);

  void ReduceScalarValue(
      const at::Scalar& input,
      const at::ScalarType type,
      uint32_t& host_ptr_offset);

  template <typename T>
  void AddAttr(const string& attr_name, T value) {
    OperatorAttrMaker::SetAttr(attr_name, value, ir_node_);
  }

  template <typename T>
  void ReduceScalarValueOp(T* value, uint32_t& host_ptr_offset) {
    ScalarMemContext::GetContext().AppendToHostMem(
        reinterpret_cast<uint8_t*>(value),
        sizeof(T),
        host_ptr_offset);
  }

  void Run();
private:
  void AddZeroDimInput(const at::Tensor& input, const string& desc_name);

  uint32_t output_index_ = 0;
  uint32_t input_index_ = 0;
  NodePtr ir_node_ = nullptr;
};
} // namespace native
} // namespace at_npu