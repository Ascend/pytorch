// Copyright (c) 2022 Huawei Technologies Co., Ltd
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
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& remainder_out_scalar_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Scalar other) {
  OpCommand cmd;
  cmd.Name("FloorMod")
        .Input(self)
        .Input(other, self.scalar_type())
        .Output(result)
        .Run();

  return result;
}

Tensor& remainder_out_scalar_npu(
    const Tensor& self,
    const Scalar other,
    Tensor& result) {
  OpPreparation::CheckOut({self}, result, self);
  remainder_out_scalar_npu_nocheck(result, self, other);
  return result;
}

Tensor& remainder_out_tensor_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  if (other.dim() == 0) {
    remainder_out_scalar_npu(self, other.item(), result);
  } else {
    OpCommand cmd;
    cmd.Name("FloorMod")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }

  return result;
}

Tensor& remainder_out_tensor_npu(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  Tensor outputTensor = CalcuOpUtil::is_scalar_wrapped_to_tensor(self) ? other : self;
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(outputTensor),
      self.scalar_type(),
      outputSize);
  remainder_out_tensor_npu_nocheck(result, self, other);

  return result;
}

Tensor remainder_tensor_npu(const Tensor& self, const Tensor& other) {
  // calculate the output size
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);
  Tensor outputTensor = isSelfWrapped ? other : self;

  auto outputSize = broadcast_ops_npu_output_size(self, other);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      outputTensor.options(),
      CalcuOpUtil::get_tensor_npu_format(outputTensor));

  // calculate the output result of the NPU
  remainder_out_tensor_npu_nocheck(result, self, other);

  return result;
}

Tensor remainder_scalar_npu(const Tensor& self, Scalar other) {
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  remainder_out_scalar_npu_nocheck(result, self, other);

  return result;
}

Tensor& remainder_tensor_npu_(Tensor& self, const Tensor& other) {
  SmallVector<Tensor, N> inputs = {other};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = remainder_out_tensor_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    remainder_out_tensor_npu_nocheck(self, self, other);
  }

  return self;
}


Tensor& remainder_scalar_npu_(Tensor& self, Scalar other) {
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = remainder_out_scalar_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    remainder_out_scalar_npu_nocheck(self, self, other);
  }
  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("remainder.Scalar_out", TORCH_FN(remainder_out_scalar_npu));
  m.impl("remainder.Tensor_out", TORCH_FN(remainder_out_tensor_npu));
  m.impl("remainder.Tensor",TORCH_FN(remainder_tensor_npu));
  m.impl("remainder.Scalar",TORCH_FN(remainder_scalar_npu));
  m.impl("remainder_.Tensor",TORCH_FN(remainder_tensor_npu_));
  m.impl("remainder_.Scalar",TORCH_FN(remainder_scalar_npu_));
}

} // namespace native
} // namespace at