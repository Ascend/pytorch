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
at::Tensor true_divide_dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);
  if (not isSelfWrapped) {
    return self;
  } else {
    return other;
  }
}

at::Tensor& true_divide_Tensor_out_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Div")
     .Input(self)
     .Input(other)
     .Output(result)
     .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::true_divide_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  at::Tensor selfTemp = self;
  at::Tensor otherTemp = other;
  if (self.scalar_type() == at::ScalarType::Int) {
    selfTemp = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
    result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Float);
  }

  if (other.scalar_type() == at::ScalarType::Int||other.scalar_type() == at::ScalarType::Bool) {
    otherTemp = NPUNativeFunctions::npu_dtype_cast(other, at::ScalarType::Float);
  }

  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(selfTemp);
  bool isOtherWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(otherTemp);
  if (isSelfWrapped && (!isOtherWrapped)) {
    selfTemp = NPUNativeFunctions::npu_dtype_cast(selfTemp, otherTemp.scalar_type());
  } else if (isOtherWrapped && (!isSelfWrapped)) {
    otherTemp = NPUNativeFunctions::npu_dtype_cast(otherTemp, selfTemp.scalar_type());
  }

  at::Tensor outputTensor = true_divide_dest_output(selfTemp, otherTemp);
  auto outputSize = broadcast_ops_npu_output_size(selfTemp, otherTemp);

  OpPreparation::CheckOut(
      {outputTensor},
      result,
      CalcuOpUtil::get_tensor_npu_format(outputTensor),
      outputTensor.scalar_type(),
      outputSize);

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    true_divide_Tensor_out_npu_nocheck(selfTemp, otherTemp, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    true_divide_Tensor_out_npu_nocheck(selfTemp, otherTemp, result);
  }

  if (self.scalar_type() == at::ScalarType::Int) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Int);
  }
  return result;
}

at::Tensor& true_divide_Scalar_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Scalar other) {
  at::Tensor selfTemp = self;
  if (self.scalar_type() == at::ScalarType::Int) {
    selfTemp = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
    result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Float);
  }

  OpCommand cmd;
  cmd.Name("Div")
     .Input(selfTemp)
     .Input(other, selfTemp.scalar_type())
     .Output(result)
     .Run();

  if (self.scalar_type() == at::ScalarType::Int) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Int);
  }
  return result;
}

at::Tensor NPUNativeFunctions::true_divide(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor selfTemp = self;
  at::Tensor otherTemp = other;
  if (self.scalar_type() == at::ScalarType::Int || self.scalar_type() == at::ScalarType::Bool) {
    selfTemp = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }
  if (other.scalar_type() == at::ScalarType::Int || other.scalar_type() == at::ScalarType::Bool) {
    otherTemp = NPUNativeFunctions::npu_dtype_cast(other, at::ScalarType::Float);
  }

  at::Tensor outputTensor = true_divide_dest_output(selfTemp, otherTemp);
  auto outputSize = broadcast_ops_npu_output_size(selfTemp, otherTemp);

  at::Tensor result = OpPreparation::ApplyTensor(outputTensor, outputSize);

  true_divide_Tensor_out_npu_nocheck(selfTemp, otherTemp, result);

  return result;
}

at::Tensor NPUNativeFunctions::true_divide(const at::Tensor& self, at::Scalar other) {
  at::Tensor selfTemp = self;
  if (self.scalar_type() == at::ScalarType::Int || self.scalar_type() == at::ScalarType::Bool) {
    selfTemp = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }

  at::Tensor result = OpPreparation::ApplyTensor(self);

  true_divide_Scalar_out_npu_nocheck(result, selfTemp, other);

  return result;
}

at::Tensor& NPUNativeFunctions::true_divide_(at::Tensor& self, const at::Tensor& other) {
  NPUNativeFunctions::true_divide_out(self, other, self);

  return self;
}

at::Tensor& NPUNativeFunctions::true_divide_(at::Tensor& self, at::Scalar other) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = true_divide_Scalar_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    true_divide_Scalar_out_npu_nocheck(self, self, other);
  }

  return self;
}
} // namespace native
} // namespace at_npu
