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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include <torch/library.h>

namespace at {
namespace native {
using namespace at::native::npu;

Tensor batch_norm_npu_(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    bool training,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});
  if (input.numel() == 0) {
    // don't return view of input, don't return empty tensor because it will
    // break gradient chain
    Tensor out = input.clone();
    if (weight.defined()) {
      out = out * weight[0];
    }

    if (bias.defined()) {
      out = out + bias[0];
    }

    return out;
  }

  return std::get<0>(at::_batch_norm_impl_index(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      training,
      momentum,
      eps,
      cudnn_enabled));
}

tuple<Tensor, Tensor, Tensor, Tensor, int64_t> _batch_norm_impl_index_npu(
    const Tensor& input,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& bias,
    const c10::optional<Tensor>& running_mean,
    const c10::optional<Tensor>& running_var,
    bool training,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  Tensor reserve = at::empty({0}, input.options().dtype(kByte));
  return std::tuple_cat(
      at::native_batch_norm(
          input,
          weight,
          bias,
          running_mean,
          running_var,
          training,
          momentum,
          eps),
      std::tuple<Tensor>(reserve),
      std::make_tuple(0));
}

tuple<Tensor, Tensor, Tensor> _batch_norm_impl_index_backward_npu(
    int64_t impl_index,
    const Tensor& input,
    const Tensor& grad_output,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& running_mean,
    const c10::optional<Tensor>&running_var,
    const c10::optional<Tensor>& save_mean,
    const c10::optional<Tensor>& save_var_transform,
    bool train,
    double epsilon,
    std::array<bool, 3> output_mask,
    const Tensor& reservedSpace) {
  return at::native_batch_norm_backward(
      grad_output,
      input,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_var_transform,
      train,
      epsilon,
      output_mask);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("batch_norm", TORCH_FN(batch_norm_npu_));
  m.impl("_batch_norm_impl_index", TORCH_FN(_batch_norm_impl_index_npu));
  m.impl("_batch_norm_impl_index_backward", TORCH_FN(_batch_norm_impl_index_backward_npu));
}
} // namespace native
} // namespace at