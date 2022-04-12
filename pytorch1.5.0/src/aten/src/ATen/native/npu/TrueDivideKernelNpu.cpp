// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

Tensor& true_div_scalar_out_nocheck_npu(const Tensor &self, const Scalar other, Tensor &result){
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

Tensor& true_div_out_npu_nocheck(const Tensor &self, const Tensor &other, Tensor &result) {
  if (other.dim() == 0) {
    true_div_scalar_out_nocheck_npu(self, other.item(), result);
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

Tensor& true_divide_out_npu(Tensor &result, const Tensor &self, const Tensor &other) {  
  Tensor selfTemp = self;
  Tensor otherTemp = other;
  if (result.scalar_type() != ScalarType::Float && result.scalar_type() != ScalarType::Half) {
    TORCH_CHECK(false, "result type Float can't be cast to the desired output type ", result.scalar_type());
  }
  if (self.scalar_type() != result.scalar_type()) {
    selfTemp = self.npu_dtype_cast(result.scalar_type());
    otherTemp = other.npu_dtype_cast(result.scalar_type());
  }  

  Tensor outputTensor = CalcuOpUtil::is_scalar_wrapped_to_tensor(selfTemp) ? otherTemp : selfTemp;
  auto outputSize = broadcast_ops_npu_output_size(selfTemp, otherTemp);
  OpPreparation::CheckOut(
      {selfTemp},
      result,
      outputTensor,
      outputSize);
  if (!NpuUtils::check_match(&result)) {
    Tensor contiguousResult = NpuUtils::format_contiguous(selfTemp);
    true_div_out_npu_nocheck(selfTemp, otherTemp, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    true_div_out_npu_nocheck(selfTemp, otherTemp, result);
  }
  return result;
}

Tensor true_divide_npu(const Tensor &self, const Tensor &other) {
  Tensor selfTemp = self;
  Tensor otherTemp = other;
  if (self.scalar_type() == ScalarType::Int || self.scalar_type() == ScalarType::Bool) {
    selfTemp = self.npu_dtype_cast(ScalarType::Float);
  }
  if (other.scalar_type() == ScalarType::Int) {
    otherTemp = other.to(ScalarType::Float);
  }
  
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(selfTemp);
  Tensor outputTensor = isSelfWrapped ? otherTemp : selfTemp;
  auto outputSize = broadcast_ops_npu_output_size(selfTemp, otherTemp);
  Tensor result = OpPreparation::ApplyTensor(outputTensor, outputSize);
  true_div_out_npu_nocheck(selfTemp, otherTemp, result);
  return result;
}

Tensor true_divide_npu(const Tensor &self, Scalar other) {
  auto outputSize = input_same_output_size(self);
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  true_div_scalar_out_nocheck_npu(self, other, result);
  return result;
}

Tensor& true_divide_npu_(Tensor &self, const Tensor &other) {
  Tensor otherTemp = other;
  if (self.scalar_type() != other.scalar_type()) {
    otherTemp = other.to(self.scalar_type());
  }
  true_divide_out_npu(self, otherTemp, self);
  return self;
}

Tensor& true_divide_npu_(Tensor &self, Scalar other) {
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    true_div_scalar_out_nocheck_npu(contiguousSelf, other, contiguousSelf);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    true_div_scalar_out_nocheck_npu(self, other, self);
  }
  return self;
}
} // namespace native
} // namespace at
