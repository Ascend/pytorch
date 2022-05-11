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
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> apply_adam_out_npu_nocheck(
    at::Scalar beta1_power,
    at::Scalar beta2_power,
    at::Scalar lr,
    at::Scalar beta1,
    at::Scalar beta2,
    at::Scalar epsilon,
    const at::Tensor& grad,
    c10::optional<bool> use_locking,
    c10::optional<bool> use_nesterov,
    at::Tensor& var_out,
    at::Tensor& m_out,
    at::Tensor& v_out) {
  OpCommand cmd;
  cmd.Name("ApplyAdamD")
     .Input(var_out)
     .Input(m_out)
     .Input(v_out)
     .Input(beta1_power, var_out.scalar_type())
     .Input(beta2_power, var_out.scalar_type())
     .Input(lr, var_out.scalar_type())
     .Input(beta1, var_out.scalar_type())
     .Input(beta2, var_out.scalar_type())
     .Input(epsilon, var_out.scalar_type())
     .Input(grad)
     .Output(var_out)
     .Output(m_out)
     .Output(v_out);
  if (use_locking != c10::nullopt) {
    cmd.Attr("use_locking", bool(use_locking));
  }
  if (use_nesterov != c10::nullopt) {
    cmd.Attr("use_nesterov", bool(use_nesterov));
  }
  cmd.Run();
  return std::tie(var_out, m_out, v_out);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::npu_apply_adam(
    at::Scalar beta1_power,
    at::Scalar beta2_power,
    at::Scalar lr,
    at::Scalar beta1,
    at::Scalar beta2,
    at::Scalar epsilon,
    const at::Tensor& grad,
    c10::optional<bool> use_locking,
    c10::optional<bool> use_nesterov) {
  AT_ERROR("npu_apply_adam is not implemented for Tensor");
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> NPUNativeFunctions::npu_apply_adam_out(
    at::Scalar beta1_power,
    at::Scalar beta2_power,
    at::Scalar lr,
    at::Scalar beta1,
    at::Scalar beta2,
    at::Scalar epsilon,
    const at::Tensor& grad,
    c10::optional<bool> use_locking,
    c10::optional<bool> use_nesterov,
    at::Tensor& var,
    at::Tensor& m,
    at::Tensor& v) {
  bool var_match = NpuUtils::check_match(&var);
  bool m_match = NpuUtils::check_match(&m);
  bool v_match = NpuUtils::check_match(&v);
  if (!(var_match && m_match && v_match)) {
    at::Tensor contiguous_var = var_match ? var : NpuUtils::format_contiguous(var);
    at::Tensor contiguous_m = m_match ? m : NpuUtils::format_contiguous(m);
    at::Tensor contiguous_v = v_match ? v : NpuUtils::format_contiguous(v);
    apply_adam_out_npu_nocheck(
        beta1_power,
        beta2_power,
        lr,
        beta1,
        beta2,
        epsilon,
        grad,
        use_locking,
        use_nesterov,
        contiguous_var,
        contiguous_m,
        contiguous_v);
    if (!var_match) {
      NpuUtils::format_fresh_view(var, contiguous_var);
    }
    if (!m_match) {
      NpuUtils::format_fresh_view(m, contiguous_m);
    }
    if (!v_match) {
      NpuUtils::format_fresh_view(v, contiguous_v);
    }
  } else {
    apply_adam_out_npu_nocheck(
        beta1_power,
        beta2_power,
        lr,
        beta1,
        beta2,
        epsilon,
        grad,
        use_locking,
        use_nesterov,
        var,
        m,
        v);
  }
  return std::tie(var, m, v);
}

} // namespace native
} // namespace at_npu
