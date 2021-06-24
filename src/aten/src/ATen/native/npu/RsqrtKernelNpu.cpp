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

Tensor& rsqrt_out_npu_nocheck(Tensor& result, const Tensor& self) {
  OpCommand cmd;
  cmd.Name("Rsqrt")
     .Input(self)
     .Output(result)
     .Run();

  return result;
}

Tensor& rsqrt_out_npu(Tensor& result, const Tensor& self) {
  OpPreparation::CheckOut({self}, result, self);
  rsqrt_out_npu_nocheck(result, self);

  return result;
}

Tensor rsqrt_npu(const Tensor& self) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  rsqrt_out_npu_nocheck(result, self);

  return result;
}

Tensor& rsqrt_npu_(Tensor& self) {
  if (!NpuUtils::check_match(&self)) {
    Tensor selfContiguous = NpuUtils::format_contiguous(self);
    Tensor result = rsqrt_out_npu_nocheck(selfContiguous, selfContiguous);
    NpuUtils::format_fresh_view(self, result);
  } else {
    rsqrt_out_npu_nocheck(self, self);
  }

  return self;
}
} // namespace native
} // namespace at