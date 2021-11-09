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

#include "ATen/native/npu/utils/NpuUtils.h"

#include <ATen/ATen.h>
#include <c10/npu/NPUGraph.h>
namespace at {
namespace native {
namespace npu {
using c10::npu::graph::DynamicInputRegFunc;
using c10::npu::graph::DyNumAndIndex;
using c10::npu::graph::NodePtr;

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
      const string& desc_name,
      MemoryType mem_type);

  void AddInput(
      const IntArrayRef& dim_list,
      const ScalarType to_type,
      const string& desc_name);

  void AddOutput(
      const at::Tensor& output,
      const string& desc_name = "",
      const string& real_type = "",
      const optional<aclFormat>& sensitive_format = nullopt);

  void AddDynamicInputRegFunc(
      DynamicInputRegFunc func,
      DyNumAndIndex num_and_index);

  template <typename T>
  void AddAttr(const string& attr_name, T value) {}

private:
  void AddZeroDimInput(const at::Tensor& input, const string& desc_name);

  uint32_t output_index_ = 0;
  uint32_t input_index_ = 0;
  NodePtr ir_node_ = nullptr;
};
} // namespace npu
} // namespace native
} // namespace at