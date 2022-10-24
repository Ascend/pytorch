// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::nll_loss2d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight,
    at::Tensor& grad_input) {
  at::Tensor weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  at::Tensor weight_tensor;
  if (weight.defined()) {
    weight_tensor = NpuUtils::format_contiguous(weight);
  } else {
    weight_tensor = at::ones(self.size(1), self.options());
  }

  if (ignore_index >= 0) {
    at::Tensor zero = at::zeros(1, self.options());
    CalcuOpUtil::AclrtMemcpyAsync(
        {weight_tensor, ignore_index},
        weight_tensor.itemsize(),
        {zero, 0},
        weight_tensor.itemsize(),
        ACL_MEMCPY_DEVICE_TO_DEVICE);
  }

  auto reductionStr = CalcuOpUtil::get_reduction_str(reduction);

  OpPreparation::CheckMemory({self, grad_output, target, weight_tensor, total_weight}, {grad_input});
  OpCommand cmd;
  cmd.Name("NLLLossGrad")
      .Input(self)
      .Input(grad_output)
      .Input(target)
      .Input(weight_tensor)
      .Input(total_weight)
      .Attr("reduction", reductionStr)
      .Output(grad_input)
      .Run();
  return grad_input;
}

at::Tensor NPUNativeFunctions::nll_loss2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight) {
  // Check Target Dtype
  auto scalar_type = target.scalar_type();
  TORCH_CHECK(scalar_type == at::kLong || scalar_type == at::kInt,
      "Expected object of scalar type ", at::kLong, " or ", at::kInt, " but got scalar type ", scalar_type,
      " for argument 'target'  in call to nll_loss2d_backward");
  at::Tensor targetCast = NPUNativeFunctions::npu_dtype_cast(target, at::kInt);

  auto self_input = self.contiguous();
  self_input = self_input.permute({0, 2, 3, 1});
  self_input = self_input.reshape({-1, self.size(1)});

  auto target_input = targetCast.contiguous();
  target_input = targetCast.reshape({-1});

  auto grad_output_reshape = grad_output.contiguous();
  if (reduction == at::Reduction::None) {
    grad_output_reshape = grad_output_reshape.reshape({-1});
  }

  auto outputSize = input_same_output_size(self_input);
  at::Tensor grad_input = OpPreparation::ApplyTensorWithFormat(
      outputSize, self_input.options(), CalcuOpUtil::get_tensor_npu_format(self_input));
  // calculate the output result of the NPU
  NPUNativeFunctions::nll_loss2d_backward_out(
      grad_output_reshape,
      self_input,
      target_input,
      weight_opt,
      reduction,
      ignore_index,
      total_weight,
      grad_input);

  grad_input =
      grad_input.reshape({self.size(0), self.size(2), self.size(3), self.size(1)});
  grad_input = grad_input.permute({0, 3, 1, 2});

  return grad_input;
}

} // namespace native
} // namespace at_npu