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
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& mse_loss_backward_out_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& grad_input) {
  if (self.numel() == 0 || target.numel() == 0) {
    grad_input = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    return grad_input;
  }

  OpPreparation::CheckMemory({grad_output, self, target}, {grad_input});
  string reductionStr(CalcuOpUtil::get_reduction_str(reduction));

  OpCommand cmd;
  cmd.Name("MseLossGrad")
      .Input(self)
      .Input(target)
      .Input(grad_output)
      .Output(grad_input)
      .Attr("reduction", reductionStr)
      .Run();
  return grad_input;
}

Tensor mse_loss_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  auto grad_out = grad_output.contiguous();
  if (grad_out.dim() == 0) {
    grad_out.view(1);
  }
  Tensor grad_input = OpPreparation::ApplyTensor(self);

  mse_loss_backward_out_npu(
      grad_out,
      self,
      target,
      reduction,
      grad_input);
  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("mse_loss_backward.grad_input", TORCH_FN(mse_loss_backward_out_npu));
  m.impl("mse_loss_backward", TORCH_FN(mse_loss_backward_npu));
}
} // namespace native
} // namespace at