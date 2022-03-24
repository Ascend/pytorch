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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
tuple<at::Tensor&, at::Tensor&, at::Tensor&> bert_apply_adam_out_npu_nocheck(
    at::Tensor& var_out,
    at::Tensor& m_out,
    at::Tensor& v_out,
    const at::Tensor& var,
    const at::Tensor& m,
    const at::Tensor& v,
    at::Scalar lr,
    at::Scalar beta1,
    at::Scalar beta2,
    at::Scalar epsilon,
    const at::Tensor& grad,
    at::Scalar max_grad_norm,
    at::Scalar global_grad_norm,
    at::Scalar weight_decay,
    c10::optional<at::Scalar> step_size,
    int64_t adam_mode) {
  std::string adamMode = adam_mode == 0 ? "adam" : "mbart_adam";
  OpCommand cmd;
  cmd.Name("ApplyAdamV2")
      .Input(var)
      .Input(m)
      .Input(v)
      .Input(lr, var.scalar_type())
      .Input(beta1, var.scalar_type())
      .Input(beta2, var.scalar_type())
      .Input(epsilon, var.scalar_type())
      .Input(grad)
      .Input(max_grad_norm, var.scalar_type())
      .Input(global_grad_norm, var.scalar_type())
      .Input(weight_decay, var.scalar_type());
  if (step_size.has_value()) {
    cmd.Input(step_size.value(), var.scalar_type());
  }
  cmd.Output(var_out)
      .Output(m_out)
      .Output(v_out)
      .Attr("adam_mode", adamMode)
      .Run();
  return std::tie(var_out, m_out, v_out);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::npu_bert_apply_adam(
    c10::Scalar lr,
    c10::Scalar beta1,
    c10::Scalar beta2,
    c10::Scalar epsilon,
    const at::Tensor& grad,
    c10::Scalar max_grad_norm,
    c10::Scalar global_grad_norm,
    c10::Scalar weight_decay,
    c10::optional<at::Scalar> step_size,
    int64_t adam_mode) {
  AT_ERROR("npu_bert_apply_adam is not implemented for Tensor");
}

tuple<at::Tensor&, at::Tensor&, at::Tensor&> NPUNativeFunctions::npu_bert_apply_adam_out(
    at::Scalar lr,
    at::Scalar beta1,
    at::Scalar beta2,
    at::Scalar epsilon,
    const at::Tensor& grad,
    at::Scalar max_grad_norm,
    at::Scalar global_grad_norm,
    at::Scalar weight_decay,
    c10::optional<at::Scalar> step_size,
    int64_t adam_mode,
    at::Tensor& var,
    at::Tensor& m,
    at::Tensor& v) {
  bert_apply_adam_out_npu_nocheck(
      var, m, v, var, m, v,
      lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size, adam_mode);

  return std::tie(var, m, v);
}

} // namespace native
} // namespace at_npu