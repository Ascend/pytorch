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
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;
std::tuple<Tensor, Tensor, Tensor> apply_adam_out_npu_nocheck(
    Tensor& var_out,
    Tensor& m_out,
    Tensor& v_out,
    const Tensor& var,
    const Tensor& m,
    const Tensor& v,
    Scalar beta1_power,
    Scalar beta2_power,
    Scalar lr,
    Scalar beta1,
    Scalar beta2,
    Scalar epsilon,
    const Tensor& grad,
    c10::optional<bool> use_locking,
    c10::optional<bool> use_nesterov) {
  Tensor var_tmp = var;
  Tensor m_tmp = m;
  Tensor v_tmp = v;
  OpCommand cmd;
  cmd.Name("ApplyAdam")
     .Input(var_tmp)
     .Input(m_tmp)
     .Input(v_tmp)
     .Input(beta1_power, var.scalar_type())
     .Input(beta2_power, var.scalar_type())
     .Input(lr, var.scalar_type())
     .Input(beta1, var.scalar_type())
     .Input(beta2, var.scalar_type())
     .Input(epsilon, var.scalar_type())
     .Input(grad)
     .Output(var_tmp);
  if (use_locking != c10::nullopt) {
    cmd.Attr("use_locking", bool(use_locking));
  }
  if (use_nesterov != c10::nullopt) {
    cmd.Attr("use_nesterov", bool(use_nesterov));
  }
  cmd.Run();

  var_out.copy_(var_tmp);
  m_out.copy_(m_tmp);
  v_out.copy_(v_tmp);
  return std::tie(var_out, m_out, v_out);
}

std::tuple<Tensor, Tensor, Tensor> apply_adam_npu(
    const Tensor& var,
    const Tensor& m,
    const Tensor& v,
    Scalar beta1_power,
    Scalar beta2_power,
    Scalar lr,
    Scalar beta1,
    Scalar beta2,
    Scalar epsilon,
    const Tensor& grad,
    c10::optional<bool> use_locking,
    c10::optional<bool> use_nesterov) {
  Tensor var_out = OpPreparation::ApplyTensor(var);
  Tensor m_out = OpPreparation::ApplyTensor(m);
  Tensor v_out = OpPreparation::ApplyTensor(v);

  bool var_match = NpuUtils::check_match(&var);
  bool m_match = NpuUtils::check_match(&m);
  bool v_match = NpuUtils::check_match(&v);
  if (!(var_match && m_match && v_match)) {
    Tensor contiguous_var = var_match ? var : NpuUtils::format_contiguous(var);
    Tensor contiguous_m = m_match ? m : NpuUtils::format_contiguous(m);
    Tensor contiguous_v = v_match ? v : NpuUtils::format_contiguous(v);
    var_out = OpPreparation::ApplyTensor(contiguous_var);
    m_out = OpPreparation::ApplyTensor(contiguous_m);
    v_out = OpPreparation::ApplyTensor(contiguous_v);
    apply_adam_out_npu_nocheck(
        var_out,
        m_out,
        v_out,
        contiguous_var,
        contiguous_m,
        contiguous_v,
        beta1_power,
        beta2_power,
        lr,
        beta1,
        beta2,
        epsilon,
        grad,
        use_locking,
        use_nesterov);
  } else {
    apply_adam_out_npu_nocheck(
        var_out,
        m_out,
        v_out,
        var,
        m,
        v,
        beta1_power,
        beta2_power,
        lr,
        beta1,
        beta2,
        epsilon,
        grad,
        use_locking,
        use_nesterov);
  }

  return std::tie(var_out, m_out, v_out);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("npu_apply_adam", TORCH_FN(apply_adam_npu));
}
} // namespace native
} // namespace at
