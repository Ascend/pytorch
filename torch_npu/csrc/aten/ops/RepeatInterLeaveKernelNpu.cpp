// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& repeat_interleave_out_npu(at::Tensor& result, const at::Tensor& self, int64_t repeats) {
  at::Scalar repeat = repeats;
  OpCommand cmd;
  cmd.Name("RepeatInterleave")
    .Input(self)
    .Input(repeat, at::kLong)
    .Output(result)
    .Attr("axis", (int64_t)0)
    .Run();

  return result;
}

at::Tensor& repeat_interleave_out_npu(at::Tensor& result, const at::Tensor& self, const at::Tensor& repeats) {
  OpCommand cmd;
  cmd.Name("RepeatInterleave")
    .Input(self)
    .Input(repeats)
    .Output(result)
    .Attr("axis", (int64_t)0)
    .Run();

  return result;
}

at::Tensor NPUNativeFunctions::repeat_interleave(
    const at::Tensor& self,
    int64_t repeats,
    c10::optional<int64_t> dim) {
  int64_t realDim = dim.value_or(0);
  int64_t self_dim = self.dim();
  TORCH_CHECK(
      (realDim >= -self_dim) && (realDim <= self_dim - 1),
      "dim value should be in the range of [-x, x-1], x is the dimension number of input tensor.");
  TORCH_CHECK(
      repeats >= 1,
      "repeats can not be negative.");
  at::Tensor selfTensor = self;
  if (!dim.has_value()) {
    selfTensor = at::flatten(selfTensor);
  }
  if (repeats == 1) {
    return selfTensor;
  }

  if (self_dim > 1 && realDim != 0) {
    selfTensor = selfTensor.transpose(0, realDim);
  }

  auto outputSize = repeat_interleave_npu_output_size(selfTensor, repeats, 0);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(selfTensor, outputSize, ACL_FORMAT_ND);
  repeat_interleave_out_npu(result, selfTensor, repeats);
  if (self_dim > 1 && realDim != 0) {
    result = result.transpose(0, realDim);
  }
  return result;
}

at::Tensor NPUNativeFunctions::repeat_interleave(
    const at::Tensor& self,
    const at::Tensor& repeats,
    c10::optional<int64_t> dim) {
  int64_t realDim = dim.value_or(0);
  int64_t self_dim = self.dim();
  TORCH_CHECK(
      (realDim >= -self_dim) && (realDim <= self_dim - 1),
      "dim value should be in the range of [-x, x-1], x is the dimension number of input tensor.");

  at::Tensor selfTensor = self;
  at::Tensor repeatsTensor = repeats;
  if (!dim.has_value()) {
    selfTensor = at::flatten(selfTensor);
  }
  if (repeats.dim() == 1 && repeats.size(0) == 1) {
    return selfTensor;
  }

  TORCH_CHECK(
      repeats.size(0) == selfTensor.size(realDim),
      "repeats must have the same size as input along dim.");

  if (self_dim > 1 && realDim != 0) {
    selfTensor = selfTensor.transpose(0, realDim);
  }

  repeatsTensor = NPUNativeFunctions::npu_dtype_cast(repeatsTensor, at::ScalarType::Int);
  repeatsTensor = NPUNativeFunctions::npu_dtype_cast(repeatsTensor, at::ScalarType::Float);
  auto outputSize = repeat_interleave_npu_output_size(selfTensor, repeatsTensor, 0);

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(selfTensor, outputSize, ACL_FORMAT_ND);
  repeat_interleave_out_npu(result, selfTensor, repeats);
  if (self_dim > 1 && realDim != 0) {
    result = result.transpose(0, realDim);
  }
  return result;
}

} // namespace native
} // namespace at_npu