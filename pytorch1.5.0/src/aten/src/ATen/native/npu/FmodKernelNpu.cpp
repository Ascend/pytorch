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

Tensor& fmod_out_npu_nocheck(Tensor& result, const Tensor& self, const Tensor& other) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  OpCommand cmd;
  cmd.Name("FloorMod")
    .Expect(unified_result)
    .Input(self)
    .Input(other)
    .Output(result)
    .Run();

  return result;
}

Tensor& fmod_out_npu_nocheck(Tensor& result, const Tensor& self, Scalar other) {
  OpCommand cmd;
  cmd.Name("FloorMod")
    .Input(self)
    .Input(other, self.scalar_type())
    .Output(result)
    .Run();

  return result;
}

Tensor& fmod_out_npu(Tensor& result, const Tensor& self, const Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
    {self, other}, 
    result,
    self, 
    outputSize);
  
  fmod_out_npu_nocheck(result, self, other);
  return result;
}

Tensor& fmod_out_npu(Tensor& result, const Tensor& self, Scalar other) {
  OpPreparation::CheckOut({self}, result, self);
  fmod_out_npu_nocheck(result, self, other);
  return result;
}

Tensor& fmod_npu_(Tensor& self, Scalar other) {
  OpPreparation::CheckMemory({self}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = fmod_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    fmod_out_npu_nocheck(self, self, other);
  }

  return self;
}

Tensor& fmod_npu_(Tensor& self, const Tensor& other) {
  OpPreparation::CheckMemory({self, other}, {self}); 
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = fmod_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    fmod_out_npu_nocheck(self, self, other);
  }

  return self;
}

Tensor fmod_npu(const Tensor& self, Scalar other) {
  Tensor result = OpPreparation::ApplyTensor(self);
  fmod_out_npu_nocheck(result, self, other);
  return result;
}

Tensor fmod_npu(const Tensor& self, const Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  fmod_out_npu_nocheck(result, self, other);
  return result;
}

} // namespace native
} // namespace at