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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/OpTemplate.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& sqrt_out_npu_safe(Tensor& result, const Tensor& self) {

  OpCommand cmd;
  cmd.Name("Sqrt")
    .Input(self)
    .Output(result)
    .Run();

  return result;
}

Tensor& sqrt_out_npu(Tensor& result, const Tensor& self) {
  OpPreparation::CheckOut({self}, result, self);
  sqrt_out_npu_safe(result, self);

  return result;
}

Tensor sqrt_npu(const Tensor& self) {
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      self.sizes(), self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  sqrt_out_npu_safe(result, self);
  return result;
}

Tensor& sqrt_npu_(Tensor& self) {
  SmallVector<Tensor, N> inputs = {self};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = sqrt_out_npu_safe(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    sqrt_out_npu_safe(self, self);
  }

  return self;
}

} // namespace native
} // namespace at