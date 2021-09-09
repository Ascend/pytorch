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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& div_rounding_compute(Tensor& result, std::string rounding_mode){
  if (rounding_mode == "trunc"){
    result.trunc_();
  } else if(rounding_mode == "floor"){
    result.floor_();
  }
  return result;
}

Tensor& div_scalar_out_npu(const Tensor& self, const Scalar other, Tensor& result) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  OpCommand cmd;
  cmd.Name("Div")
        .Expect(unified_result)
        .Input(self)
        .Input(other, self.scalar_type())
        .Output(result)
        .Run();
  
  return result;
}

Tensor& div_scalar_out_npu(const Tensor& self, const Scalar other, std::string rounding_mode, Tensor& result) {
  div_scalar_out_npu(self, other, result);
  div_rounding_compute(result, rounding_mode);
  return result;
}

Tensor& div_out_npu_nocheck(const Tensor& self, const Tensor& other, Tensor& result) {

  // executing the NPU operator
  if (other.dim() == 0) {
    div_scalar_out_npu(self, other.item(), result);
  } else {
    auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
    OpCommand cmd;
    cmd.Name("Div")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();    
  }

  return result;
}

Tensor& div_out_npu(const Tensor& self, const Tensor& other, Tensor& result) {
  // calculate the output size
  Tensor outputTensor = CalcuOpUtil::is_scalar_wrapped_to_tensor(self) ? other : self;
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
    {self},
    result, 
    CalcuOpUtil::get_tensor_npu_format(outputTensor),
    self.scalar_type(), 
    outputSize);
  div_out_npu_nocheck(self, other, result);

  return result;
}

Tensor& div_outmode_out_npu(const Tensor& self, const Tensor& other, std::string rounding_mode, Tensor& result){
  div_out_npu(self, other, result);
  div_rounding_compute(result, rounding_mode);
  return result;
}

Tensor div_npu(const Tensor& self, const Tensor& other) {
  // calculate the output size
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);
  Tensor outputTensor = isSelfWrapped ? other : self;

  auto outputSize = broadcast_ops_npu_output_size(self, other);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize,
      outputTensor.options(),
      CalcuOpUtil::get_tensor_npu_format(outputTensor));

  // calculate the output result of the NPU
  div_out_npu_nocheck(self, other, result);

  return result;
}

Tensor div_tensor_mode_npu(const Tensor& self, const Tensor& other, std::string rounding_mode) {
  // calculate the output size
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);
  Tensor outputTensor = isSelfWrapped ? other : self;

  auto outputSize = broadcast_ops_npu_output_size(self, other);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize,
      outputTensor.options(),
      CalcuOpUtil::get_tensor_npu_format(outputTensor));

  // calculate the output result of the NPU
  div_outmode_out_npu(self, other, rounding_mode, result);

  return result;
}

Tensor div_scalar_npu(const Tensor& self, Scalar other) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, 
      self.options(), 
      CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  div_scalar_out_npu(self, other, result);

  return result;
}

Tensor div_scalar_mode_npu(const Tensor& self, Scalar other, std::string rounding_mode){
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, 
      self.options(), 
      CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  div_scalar_out_npu(self, other, rounding_mode, result);
  return result;
}

Tensor& div_npu_(Tensor& self, const Tensor& other) {
  SmallVector<Tensor, N> inputs = {self, other};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = div_out_npu_nocheck(contiguousSelf, other, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    div_out_npu_nocheck(self, other, self);
  }

  return self;
}

Tensor& div_tensor_mode_npu_(Tensor& self, const Tensor& other, std::string rounding_mode){
  div_npu_(self, other);
  div_rounding_compute(self, rounding_mode);
  return self;
}

Tensor& div_scalar_npu_(Tensor& self, Scalar other) {
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    div_scalar_out_npu(contiguousSelf, other, contiguousSelf);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    div_scalar_out_npu(self, other, self);
  }
  return self;
}

Tensor& div_scalar_mode_npu_(Tensor& self, Scalar other, std::string rounding_mode){
  div_scalar_npu_(self, other);
  div_rounding_compute(self, rounding_mode);
  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("div.Tensor", TORCH_FN(div_npu));
  m.impl("div_.Tensor", TORCH_FN(div_npu_));
  m.impl("div.out", TORCH_FN(div_out_npu));
  m.impl("div.Tensor_mode", TORCH_FN(div_tensor_mode_npu));
  m.impl("div_.Tensor_mode", TORCH_FN(div_tensor_mode_npu_));
  m.impl("div.out_mode", TORCH_FN(div_outmode_out_npu));
  m.impl("div.Scalar", TORCH_FN(div_scalar_npu));
  m.impl("div_.Scalar", TORCH_FN(div_scalar_npu_));
  m.impl("div.Scalar_mode", TORCH_FN(div_scalar_mode_npu));
  m.impl("div_.Scalar_mode", TORCH_FN(div_scalar_mode_npu_));
} 

} // namespace native
} // namespace at