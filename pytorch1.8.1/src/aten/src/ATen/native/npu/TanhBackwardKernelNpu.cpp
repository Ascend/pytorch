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

Tensor& tanh_backward_out_npu_nocheck(
    Tensor& result,
    const Tensor& grad_output,
    const Tensor& self) {
  OpCommand cmd;
  cmd.Name("TanhGrad")
    .Input(self)
    .Input(grad_output)
    .Output(result)
    .Run();

  return result;
}

Tensor& tanh_backward_out_npu(
    const Tensor& grad_output,
    const Tensor& self,
    Tensor& result) {
  OpPreparation::CheckOut({grad_output, self}, result, self);
  tanh_backward_out_npu_nocheck(result, grad_output, self);
  return result;
}

Tensor tanh_backward_npu(const Tensor& grad_output, const Tensor& self) {
  Tensor result = OpPreparation::ApplyTensor(self);
  tanh_backward_out_npu_nocheck(result, grad_output, self);

  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("tanh_backward", TORCH_FN(tanh_backward_npu));
  m.impl("tanh_backward.grad_input", TORCH_FN(tanh_backward_out_npu));
}

} // namespace native
} // namespace at
