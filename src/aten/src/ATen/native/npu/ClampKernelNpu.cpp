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

#include <climits>
#include <float.h>
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& clamp_min_out_npu_nocheck(
    Tensor& result, 
    const Tensor& self, 
    Scalar min) {
  // Set max according to self.dtype()
  Scalar max;
  if (self.dtype() == at::kInt) {
    max = INT_MAX;

  } else if (self.dtype() == at::kFloat) {
    max = FLT_MAX;
    
  } else {
    max = NPU_HALF_MAX;
  }

  Tensor minTensor = CalcuOpUtil::CopyScalarToDevice(min, self.scalar_type());
  Tensor maxTensor = CalcuOpUtil::CopyScalarToDevice(max, self.scalar_type());
  
  OpCommand cmd;
  cmd.Name("ClipByValue")
      .Input(self)
      .Input(minTensor)
      .Input(maxTensor)
      .Output(result)
      .Run();
  return result;
}

Tensor& clamp_min_out_npu(
    Tensor& result, 
    const Tensor& self, 
    Scalar min) {
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      self.scalar_type(),
      self.sizes());

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self, &min](Tensor& result){clamp_min_out_npu_nocheck(result, self, min);})
   .Call(result);

  return result;
}

Tensor& clamp_max_out_npu(Tensor& result, const Tensor& self, Scalar max) {
  // Set min according to self.dtype()
  Scalar min;
  if (self.dtype() == at::kInt) {
    min = INT_MIN;

  } else if (self.dtype() == at::kFloat) {
    min = -FLT_MAX;
    
  } else {
    min = NPU_HALF_MIN;
  }
  
  Tensor minTensor = CalcuOpUtil::CopyScalarToDevice(min, self.scalar_type());
  Tensor maxTensor = CalcuOpUtil::CopyScalarToDevice(max, self.scalar_type());
  
  OpCommand cmd;
  cmd.Name("ClipByValue")
      .Input(self)
      .Input(minTensor)
      .Input(maxTensor)
      .Output(result)
      .Run();
  return result;
}

Tensor& clamp_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    optional<Scalar> min,
    optional<Scalar> max) {
  if (!min.has_value()) {
    Scalar maxScalar = max.value();
    clamp_max_out_npu(result, self, maxScalar);

  } else if (!max.has_value()) {
    Scalar minScalar = min.value();
    clamp_min_out_npu(result, self, minScalar);

  } else {   
    Tensor minTensor = CalcuOpUtil::CopyScalarToDevice(min.value(), self.scalar_type());
    Tensor maxTensor = CalcuOpUtil::CopyScalarToDevice(max.value(), self.scalar_type());
    
    OpCommand cmd;
    cmd.Name("ClipByValue")
        .Input(self)
        .Input(minTensor)
        .Input(maxTensor)
        .Output(result)
        .Run();   
  }

  return result;
}

Tensor& clamp_out_npu(
    Tensor& result,
    const Tensor& self,
    optional<Scalar> min,
    optional<Scalar> max) {
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      self.scalar_type(),
      self.sizes());

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self, &min, &max](Tensor& result){clamp_out_npu_nocheck(result, self, min, max);})
   .Call(result);

  return result;
}

Tensor clamp_min_npu(const Tensor& self, Scalar min) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  clamp_min_out_npu_nocheck(result, self, min);

  return result;
}

Tensor& clamp_min_npu_(Tensor& self, Scalar min) {
  clamp_min_out_npu(self, self, min);

  return self;
}

Tensor clamp_max_npu(const Tensor& self, Scalar max) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  clamp_max_out_npu(result, self, max);

  return result;
}

Tensor& clamp_max_npu_(Tensor& self, Scalar max) {
  SmallVector<Tensor, N> inputs = {self};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = clamp_max_out_npu(contiguousSelf, contiguousSelf, max);
    NpuUtils::format_fresh_view(self, result);
  } else {
    clamp_max_out_npu(self, self, max);
  }

  return self;
}

Tensor clamp_npu(
    const Tensor& self,
    optional<Scalar> min,
    optional<Scalar> max) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  clamp_out_npu_nocheck(result, self, min, max);

  return result;
}

Tensor& clamp_npu_(Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  clamp_out_npu(self, self, min, max);

  return self;
}

} // namespace native
} // namespace at
