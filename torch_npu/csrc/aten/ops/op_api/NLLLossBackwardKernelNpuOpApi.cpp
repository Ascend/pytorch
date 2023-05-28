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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::nll_loss_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight,
    at::Tensor& grad_input) {
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  at::Tensor weight_tensor;
  if (weight.defined()) {
    weight_tensor = NpuUtils::format_contiguous(weight);
  } else {
    weight_tensor = at::ones(self.size(-1), self.options());
  }

  if (ignore_index >= 0 && ignore_index < self.size(-1)) {
    at::Tensor zero = at::zeros(1, self.options());
    if (c10_npu::NpuRunMode::IsGraphMode()) {
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

  EXEC_NPU_CMD(aclnnNLLLossBackward, grad_output, self, target, weight_tensor, reduction, ignore_index, total_weight, grad_input);
  return grad_input;
}

at::Tensor NPUNativeOpApiFunctions::nll_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight) {
  // construct the output tensor of the NPU
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  NPUNativeOpApiFunctions::nll_loss_backward_out(
      grad_output,
      self,
      target,
      weight,
      reduction,
      ignore_index,
      total_weight,
      grad_input);

  return grad_input;
}

} // namespace native
} // namespace at_npu