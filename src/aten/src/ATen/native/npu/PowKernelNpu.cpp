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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& pow_out_npu(Tensor& result, const Tensor& self, const Tensor& exp) {
  OpCommand cmd;
  cmd.Name("Pow")
        .Input(self)
        .Input(exp)
        .Output(result)
        .Run();

  return result;
}

Tensor& pow_out_npu(Tensor& result, const Tensor& self, Scalar exp) {
  OpCommand cmd;
  cmd.Name("Pow")
     .Input(self)
     .Input(exp,self.scalar_type())
     .Output(result)
     .Run();

  return result;
}

Tensor& pow_out_npu(Tensor& result, Scalar self, const Tensor& exp) {
  OpCommand cmd;
  cmd.Name("Pow")
     .Input(self,exp.scalar_type())
     .Input(exp)
     .Output(result)
     .Run();

  return result;
}

Tensor pow_npu(const Tensor& self, const Tensor& exp) {
  // calculate the output size
  auto outputSize = broadcast_ops_npu_output_size(self, exp);
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  pow_out_npu(result, self, exp);
  return result;
}

Tensor pow_npu(const Tensor& self, Scalar exp) {
  Tensor result = OpPreparation::ApplyTensor(self);
  pow_out_npu(result, self, exp);
  return result;
}

Tensor pow_npu(Scalar self, const Tensor& exp) {
  // calculate the output size
  auto outputSize = input_same_output_size(exp);
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(outputSize, exp.options());
  
  // calculate the output result of the NPU
  pow_out_npu(result, self, exp);
  return result;
}

Tensor& pow_npu_(Tensor& self, const Tensor& exp) {
  OpPreparation::CheckMemory({self, exp}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    pow_out_npu(contiguousSelf, contiguousSelf, exp);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    pow_out_npu(self, self, exp);
  }

  return self;
}

Tensor& pow_npu_(Tensor& self, Scalar exp) {
  OpPreparation::CheckMemory({self}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    pow_out_npu(contiguousSelf, contiguousSelf, exp);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    pow_out_npu(self, self, exp);
  }

  return self;
}

} // namespace native
} // namespace at
