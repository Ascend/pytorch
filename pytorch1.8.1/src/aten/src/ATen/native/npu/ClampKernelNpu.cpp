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
  OpCommand cmd;
  cmd.Name("ClipByValue")
      .Input(self)
      .Input(min, self.scalar_type())
      .Input(max, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

Tensor& clamp_min_out_npu(
    const Tensor& self, 
    Scalar min,
    Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self, &min](Tensor& result){clamp_min_out_npu_nocheck(result, self, min);})
   .Call(result);
  return result;
}

Tensor& clamp_max_out_npu(const Tensor& self, Scalar max, Tensor& result) {
  // Set min according to self.dtype()
  Scalar min;
  if (self.dtype() == at::kInt) {
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

Tensor& clamp_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    optional<Scalar> min,
    optional<Scalar> max) {
  if (!min.has_value()) {
    Scalar maxScalar = max.value();
    clamp_max_out_npu(self, maxScalar, result);
  } else if (!max.has_value()) {
    Scalar minScalar = min.value();
    clamp_min_out_npu(self, minScalar, result);
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

Tensor& clamp_out_npu(
    const Tensor& self,
    optional<Scalar> min,
    optional<Scalar> max,
    Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self, &min, &max](Tensor& result){clamp_out_npu_nocheck(result, self, min, max);})
   .Call(result);
  return result;
}

Tensor clamp_min_npu(const Tensor& self, Scalar min) {
  Tensor result = OpPreparation::ApplyTensor(self);
  clamp_min_out_npu_nocheck(result, self, min);
  return result;
}

Tensor& clamp_min_npu_(Tensor& self, Scalar min) {
  clamp_min_out_npu(self, min, self);
  return self;
}

Tensor clamp_max_npu(const Tensor& self, Scalar max) {
  Tensor result = OpPreparation::ApplyTensor(self);
  clamp_max_out_npu(self, max, result);
  return result;
}

Tensor& clamp_max_npu_(Tensor& self, Scalar max) {
  OpPreparation::CheckMemory({self}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = clamp_max_out_npu(contiguousSelf, max, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    clamp_max_out_npu(self, max, self);
  }
  return self;
}

Tensor clamp_npu(
    const Tensor& self,
    optional<Scalar> min,
    optional<Scalar> max) {
  Tensor result = OpPreparation::ApplyTensor(self);
  clamp_out_npu_nocheck(result, self, min, max);
  return result;
}

Tensor& clamp_npu_(Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  clamp_out_npu(self, min, max, self);
  return self;
}
TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("clamp_", TORCH_FN(clamp_npu_));
  m.impl("clamp", TORCH_FN(clamp_npu));
  m.impl("clamp_max.out", TORCH_FN(clamp_max_out_npu));
  m.impl("clamp_max_", TORCH_FN(clamp_max_npu_));
  m.impl("clamp.out", TORCH_FN(clamp_out_npu));
  m.impl("clamp_max", TORCH_FN(clamp_max_npu));
  m.impl("clamp_min", TORCH_FN(clamp_min_npu));
}
} // namespace native
} // namespace at
