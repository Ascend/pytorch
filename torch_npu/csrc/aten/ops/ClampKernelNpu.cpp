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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& clamp_min_out_npu_nocheck(
    at::Tensor& result, 
    const at::Tensor& self, 
    at::Scalar min) {
  // Set max according to self.dtype()
  at::Scalar max;
  if (self.dtype() == at::kInt || self.dtype() == at::kLong) {
    max = INT_MAX;
  } else if (self.dtype() == at::kFloat) {
    max = FLT_MAX;
  } else {
    max = NPU_HALF_MAX;
  }
  OpCommand cmd;
  cmd.Name("ClipByValue")
      .Input(self)
      .Input(min, self.scalar_type())
      .Input(max, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::clamp_min_out(
    const at::Tensor& self, 
    const at::Scalar& min,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self, &min](at::Tensor& result){clamp_min_out_npu_nocheck(result, self, min);})
   .Call(result);
  return result;
}

at::Tensor& NPUNativeFunctions::clamp_max_out(const at::Tensor& self, const at::Scalar& max, at::Tensor& result) {
  // Set min according to self.dtype()
  at::Scalar min;
  if (self.dtype() == at::kInt || self.dtype() == at::kLong) {
    min = INT_MIN;
  } else if (self.dtype() == at::kFloat) {
    min = -FLT_MAX;    
  } else {
    min = NPU_HALF_MIN;
  }
  OpCommand cmd;
  cmd.Name("ClipByValue")
      .Input(self)
      .Input(min, self.scalar_type())
      .Input(max, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor& clamp_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    c10::optional<at::Scalar> min,
    c10::optional<at::Scalar> max) {
  if (!min.has_value()) {
    at::Scalar maxScalar = max.value();
    NPUNativeFunctions::clamp_max_out(self, maxScalar, result);
  } else if (!max.has_value()) {
    at::Scalar minScalar = min.value();
    NPUNativeFunctions::clamp_min_out(self, minScalar, result);
  } else {
    OpCommand cmd;
    cmd.Name("ClipByValue")
        .Input(self)
        .Input(min.value(), self.scalar_type())
        .Input(max.value(), self.scalar_type())
        .Output(result)
        .Run();   
  }
  return result;
}

at::Tensor& NPUNativeFunctions::clamp_out(
    const at::Tensor& self,
    const c10::optional<at::Scalar>& min,
    const c10::optional<at::Scalar>& max,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self, &min, &max](at::Tensor& result){clamp_out_npu_nocheck(result, self, min, max);})
   .Call(result);
  return result;
}

at::Tensor NPUNativeFunctions::clamp_min(const at::Tensor& self, const at::Scalar& min) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  clamp_min_out_npu_nocheck(result, self, min);
  return result;
}

at::Tensor& NPUNativeFunctions::clamp_min_(at::Tensor& self, const at::Scalar& min) {
  NPUNativeFunctions::clamp_min_out(self, min, self);
  return self;
}

at::Tensor NPUNativeFunctions::clamp_max(const at::Tensor& self, const at::Scalar& max) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  NPUNativeFunctions::clamp_max_out(self, max, result);
  return result;
}

at::Tensor& NPUNativeFunctions::clamp_max_(at::Tensor& self, const at::Scalar& max) {
  OpPreparation::CheckMemory({self}, {self});
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = NPUNativeFunctions::clamp_max_out(contiguousSelf, max, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    NPUNativeFunctions::clamp_max_out(self, max, self);
  }
  return self;
}

at::Tensor NPUNativeFunctions::clamp(
    const at::Tensor& self,
    const c10::optional<at::Scalar>& min,
    const c10::optional<at::Scalar>& max) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  clamp_out_npu_nocheck(result, self, min, max);
  return result;
}

at::Tensor& NPUNativeFunctions::clamp_(at::Tensor& self, const c10::optional<at::Scalar>& min, const c10::optional<at::Scalar>& max) {
  NPUNativeFunctions::clamp_out(self, min, max, self);
  return self;
}
} // namespace native
} // namespace at_npu
