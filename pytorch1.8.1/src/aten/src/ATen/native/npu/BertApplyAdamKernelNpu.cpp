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

tuple<Tensor, Tensor, Tensor> bert_apply_adam_out_npu_nocheck(
    Tensor& var_out,
    Tensor& m_out,
    Tensor& v_out,
    const Tensor& var,
    const Tensor& m,
    const Tensor& v,
    Scalar lr,
    Scalar beta1,
    Scalar beta2,
    Scalar epsilon,
    const Tensor& grad,
    Scalar max_grad_norm,
    Scalar global_grad_norm,
    Scalar weight_decay) {
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
      .Input(weight_decay, var.scalar_type())
      .Output(var_out)
      .Output(m_out)
      .Output(v_out)
      .Run();

  return std::tie(var_out, m_out, v_out);
}

tuple<Tensor, Tensor, Tensor> bert_apply_adam_npu(
    const Tensor& var,
    const Tensor& m,
    const Tensor& v,
    Scalar lr,
    Scalar beta1,
    Scalar beta2,
    Scalar epsilon,
    const Tensor& grad,
    Scalar max_grad_norm,
    Scalar global_grad_norm,
    Scalar weight_decay) {
  Tensor var_out = OpPreparation::ApplyTensor(var);
  Tensor m_out = OpPreparation::ApplyTensor(m);
  Tensor v_out = OpPreparation::ApplyTensor(v);

  bert_apply_adam_out_npu_nocheck(
      var_out, m_out, v_out, var, m, v,
      lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay);

  return std::tie(var, m, v);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("npu_bert_apply_adam", TORCH_FN(bert_apply_adam_npu));
}
} // namespace native
} // namespace at
