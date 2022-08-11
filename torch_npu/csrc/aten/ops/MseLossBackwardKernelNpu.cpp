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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/XLANativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& XLANativeFunctions::mse_loss_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& grad_input) {
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

at::Tensor XLANativeFunctions::mse_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  auto grad_out = grad_output.contiguous();
  if (grad_out.dim() == 0) {
    grad_out.view(1);
  }
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);

  XLANativeFunctions::mse_loss_backward_out(
      grad_out,
      self,
      target,
      reduction,
      grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu