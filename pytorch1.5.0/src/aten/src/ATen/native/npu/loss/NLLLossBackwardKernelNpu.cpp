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
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& nll_loss_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  Tensor weight_tensor;
  if (weight.defined()) {
    weight_tensor = NpuUtils::format_contiguous(weight);
  } else {
    weight_tensor = at::ones(self.size(1), self.options());
  }

  if (ignore_index >= 0 && ignore_index < self.size(-1)) {
    Tensor zero = at::zeros(1, self.options());
    if (c10::npu::NpuRunMode::IsGraphMode()) {
      auto ignore_tensor = weight_tensor
          .view({-1})
          .slice(0, ignore_index, ignore_index + 1, 1);
      ignore_tensor.copy_(zero);
    } else {
      CalcuOpUtil::AclrtMemcpyAsync(
          {weight_tensor, ignore_index},
          weight_tensor.itemsize(),
          {zero, 0},
          weight_tensor.itemsize(),
          ACL_MEMCPY_DEVICE_TO_DEVICE);
    }
  }

  std::string reductionStr = NpuUtils::get_reduction_str(reduction);

  Tensor targetCast = target;
  auto scalar_type = target.scalar_type();
  if (scalar_type == at::kLong) {
    targetCast = target.to(at::kInt);
  }  else if (scalar_type == at::kInt) {
    ;
  } 
  else {
    AT_ERROR("Expected object of scalar type ", at::kLong, " or ", at::kInt, " but got scalar type ", scalar_type,
        " for argument 'target'  in call to nll_loss_backward");
  }
  
  OpCommand cmd;
  cmd.Name("NLLLossGrad")
      .Input(self)
      .Input(grad_output)
      .Input(targetCast)
      .Input(weight_tensor)
      .Input(total_weight)
      .Output(grad_input)
      .Attr("reduction", reductionStr)
      .Attr("ignore_index", ignore_index)
      .Run();

  return grad_input;
}

Tensor nll_loss_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor grad_input = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  nll_loss_backward_out_npu(
      grad_input,
      grad_output,
      self,
      target,
      weight,
      reduction,
      ignore_index,
      total_weight);

  return grad_input;
}

} // namespace native
} // namespace at