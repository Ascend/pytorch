// Copyright (c) 2022 Huawei Technologies Co., Ltd
// Copyright (c) 2022, Facebook CORPORATION.
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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& xlogy_out_npu_nocheck(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Xlogy")
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& xlogy_out_npu_nocheck(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Xlogy")
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor& xlogy_out_npu_nocheck(const at::Scalar& self, const at::Tensor& other, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Xlogy")
      .Input(self, other.scalar_type())
      .Input(other)
      .Output(result)
      .Run();
  return result;
}


at::Tensor& NPUNativeFunctions::xlogy_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
    at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
    at::Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
    auto outputSize = broadcast_ops_npu_output_size(self, other);
    OpPreparation::CheckOut(
        {self, other},
        result,
        CalcuOpUtil::get_tensor_npu_format(formatCastOfSelf),
        result.scalar_type(),
        outputSize);
    xlogy_out_npu_nocheck(formatCastOfSelf, formatCastOfOther, result);
    return result;
}

at::Tensor& NPUNativeFunctions::xlogy_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      self.scalar_type(),
      self.sizes());
  xlogy_out_npu_nocheck(self, other, result);
  return result;
}

at::Tensor& NPUNativeFunctions::xlogy_out(const at::Scalar& self, const at::Tensor& other, at::Tensor& result) {
   OpPreparation::CheckOut(
       {other},
       result,
       CalcuOpUtil::get_tensor_npu_format(other),
       other.scalar_type(),
       other.sizes());
  xlogy_out_npu_nocheck(self, other, result);
  return result;
}

at::Tensor NPUNativeFunctions::xlogy(const at::Tensor& self, const at::Tensor& other) {
    bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);
    at::Tensor outputTensor = isSelfWrapped ? other : self;
    // calculate the output size
    auto outputSize = broadcast_ops_npu_output_size(self, other);
    // construct the output tensor of the NPU
    at::Tensor result = OpPreparation::ApplyTensorWithFormat(
        outputSize,
        outputTensor.options(),
        CalcuOpUtil::get_tensor_npu_format(outputTensor));
    // calculate the output result of the NPU
    xlogy_out_npu_nocheck(self, other, result);
    return result;
}

at::Tensor NPUNativeFunctions::xlogy(const at::Tensor& self, const at::Scalar& other) {
    at::Tensor result = OpPreparation::ApplyTensor(self);
    xlogy_out_npu_nocheck(self, other, result);
    return result;

}

at::Tensor NPUNativeFunctions::xlogy(const at::Scalar& self, const at::Tensor& other) {
    at::Tensor result = OpPreparation::ApplyTensor(other);
    xlogy_out_npu_nocheck(self, other, result);
    return result;
}


at::Tensor& NPUNativeFunctions::xlogy_(at::Tensor& self, const at::Tensor& other) {
    c10::SmallVector<at::Tensor, N> inputs = {self, other};
    c10::SmallVector<at::Tensor, N> outputs = {self};
    CalcuOpUtil::check_memory_over_laps(inputs, outputs);
    if (!NpuUtils::check_match(&self))
    {
      at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
      at::Tensor result = xlogy_out_npu_nocheck(contiguousSelf, other, contiguousSelf);
      NpuUtils::format_fresh_view(self, result);
    }
    else
    {
      xlogy_out_npu_nocheck(self, other, self);
    }
    return self;
}

at::Tensor& NPUNativeFunctions::xlogy_(at::Tensor& self, const at::Scalar& other) {
    if (!NpuUtils::check_match(&self))
    {
      at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
      at::Tensor result = xlogy_out_npu_nocheck(contiguousSelf, other, result);
      NpuUtils::format_fresh_view(self, result);
    }
    else
    {
      xlogy_out_npu_nocheck(self, other, self);
    }
    return self;
}
} // namespace native
} // namespace at_npu
