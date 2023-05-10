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

#include <ATen/Tensor.h>
#include <c10/util/SmallVector.h>

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"


namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::batch_norm(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool training,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  const at::Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return at::Tensor();});
  const at::Tensor& running_var = c10::value_or_else(running_var_opt, [] {return at::Tensor();});
  if (input.numel() == 0) {
    // don't return view of input, don't return empty tensor because it will
    // break gradient chain
    at::Tensor out = input.clone();
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

tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t> NPUNativeFunctions::_batch_norm_impl_index(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    bool training,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  at::Tensor reserve = at::empty({0}, input.options().dtype(at::kByte));
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
      std::tuple<at::Tensor>(reserve),
      std::make_tuple(0));
}

tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::_batch_norm_impl_index_backward(
    int64_t impl_index,
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>&running_var,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_var_transform,
    bool train,
    double epsilon,
    std::array<bool, 3> output_mask,
    const at::Tensor& reservedSpace) {
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

} // namespace native
} // namespace at_npu