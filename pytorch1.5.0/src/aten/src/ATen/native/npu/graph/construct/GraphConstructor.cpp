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

#include "GraphConstructor.h"
#include "c10/npu/NPUCachingAllocator.h"
#include "ATen/native/npu/graph/util/GraphUtils.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
namespace npu {
using c10::npu::graph::NodeExtInfoType;
void at::native::npu::GraphCommandImpl::SetName(const std::string& name) {
  ir_node_ = std::make_shared<c10::npu::graph::Node>(name);
}

void at::native::npu::GraphCommandImpl::AddInput() {
  ++input_index_;
}

void GraphCommandImpl::AddInput(
    const Tensor& input,
    const string& desc_name,
    const string& real_dtype,
    const optional<aclFormat>& sensitive_format) {
  if (input.dim() == 0 && !input.is_npu()) {
    AddZeroDimInput(input, desc_name);
  }
  if (GraphUtils::IsTensorWithoutNode(input)) {
    if (!input.storage().data()) {
      auto storage_impl = input.storage().unsafeGetStorageImpl();
      size_t n_bytes =
          prod_intlist(storage_impl->get_npu_desc().storage_sizes_) *
          storage_impl->itemsize();
      auto data_ptr = c10::npu::NPUCachingAllocator::get()->allocate(n_bytes);
      storage_impl->set_data_ptr(std::move(data_ptr));
    }
    GraphUtils::SetDataOp(input.storage().unsafeGetStorageImpl());
  }
  if (GraphUtils::IsDataTensor(input)) {
    GraphUtils::RetainGraphDataTensor(input);
  }
  if (sensitive_format.has_value()) {
    ir_node_->AddExtInfo(
        NodeExtInfoType::SENSITIVE_FORMAT_OF_INPUT,
        std::make_pair(desc_name, sensitive_format.value()));
  }

  auto& cur_ir_value = GraphUtils::GetTensorIrValue(input);
  if (!real_dtype.empty()) {
    cur_ir_value.SetRealType(real_dtype);
  }
  ir_node_->AddInput(
      input_index_++, cur_ir_value.GetCurNode(), cur_ir_value.GetValueIndex());
  ir_node_->UpdateNodeHash(GraphUtils::GetTensorIrValueHash(input), real_dtype);
}

void GraphCommandImpl::AddInput(
    const Scalar& input,
    const ScalarType type,
    const string& desc_name,
    MemoryType mem_type) {
  ir_node_->AddExtInfo(
      NodeExtInfoType::INPUT_TYPE_SCALAR,
      std::make_tuple(input_index_++, input, type));
  ir_node_->UpdateNodeHash(CalcuOpUtil::get_scalar_float_value(input), type);
}

void GraphCommandImpl::AddInput(
    const IntArrayRef& dim_list,
    const ScalarType to_type,
    const string& desc_name) {
  vector<int64_t> val(dim_list.begin(), dim_list.end());
  ir_node_->AddExtInfo(
      NodeExtInfoType::INPUT_TYPE_LIST_LONG,
      std::make_tuple(input_index_++, std::move(val), to_type));
  ir_node_->UpdateNodeHash(dim_list, to_type);
}

void GraphCommandImpl::AddOutput(
    const Tensor& output,
    const string& desc_name,
    const string& real_type,
    const optional<aclFormat>& sensitive_format) {
  if (sensitive_format.has_value()) {
    ir_node_->AddExtInfo(
        NodeExtInfoType::SENSITIVE_FORMAT_OF_OUTPUT,
        std::make_pair(desc_name, sensitive_format.value()));
  }
  if (!ir_node_->GetInputs().empty() || output_index_ != 0) {
    Value value{ir_node_, output_index_++};
    if (!real_type.empty()) {
      value.SetRealType(real_type);
    }
    GraphUtils::SetTensorIrValue(output, value);
  } else {
    // op without input and has outputs should be treated as graph input
    GraphUtils::SetTensorIrValue(
        output, Value(ir_node_, ir_node_, output_index_++));
    GraphUtils::RetainGraphDataTensor(output);
  }
}

void GraphCommandImpl::AddDynamicInputRegFunc(
    DynamicInputRegFunc func,
    DyNumAndIndex num_and_index) {
  ir_node_->AddExtInfo(
      NodeExtInfoType::DYNAMIC_INPUT_FUNC, std::make_pair(func, num_and_index));
}

void GraphCommandImpl::AddZeroDimInput(
    const Tensor& input,
    const string& desc_name) {
  ScalarType dtype = ScalarType::Undefined;
  if (!input.unsafeGetTensorImpl()->is_wrapped_number()) {
    dtype = input.scalar_type();
  }
  TORCH_CHECK(
      dtype != ScalarType::Undefined, "Cpu tensor scalar type is undefined");
  Scalar expect_scalar = CalcuOpUtil::ConvertTensorToScalar(input);
  ir_node_->AddExtInfo(
      NodeExtInfoType::INPUT_TYPE_SCALAR,
      std::make_tuple(input_index_++, expect_scalar, dtype));
  ir_node_->UpdateNodeHash(
      CalcuOpUtil::get_scalar_float_value(expect_scalar), dtype);
}

} // namespace npu
} // namespace native
} // namespace at
