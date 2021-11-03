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

Tensor batch_norm_backward_elemt_npu_nocheck(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const Tensor& weight,
    const Tensor& mean_dy,
    const Tensor& mean_dy_xmu,
    Tensor& grad_input) {
  OpCommand cmd;
  cmd.Name("SyncBatchNormBackwardElemt")
      .Input(grad_out)
      .Input(input)
      .Input(mean)
      .Input(invstd)
      .Input(weight)
      .Input(mean_dy)
      .Input(mean_dy_xmu)
      .Output(grad_input)
      .Run();
  return grad_input;
}

void batch_norm_backward_elemt_npu_expand_tensor(
    Tensor& expand_tensor,
    size_t dim_c,
    int64_t input_ndim,
    IntArrayRef input_shape) {
  if (input_ndim >2) {
    expand_tensor = at::npu_broadcast(expand_tensor, {1, dim_c}).t();
    for (int64_t i = 0; i < input_ndim - 3; i++) {
      expand_tensor = expand_tensor.unsqueeze(1);
    }
  }
  expand_tensor = at::npu_broadcast(expand_tensor, input_shape);
}

Tensor batch_norm_backward_elemt_npu(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const Tensor& weight,
    const Tensor& mean_dy,
    const Tensor& mean_dy_xmu) {
  int64_t input_ndim = input.dim();

  TORCH_CHECK(input_ndim > 1, "input.dim() <= 1")
  size_t dim_c = input.size(1);
  IntArrayRef input_shape = input.sizes();
  Tensor mean_expanded(mean);

  batch_norm_backward_elemt_npu_expand_tensor(mean_expanded, dim_c, input_ndim, input_shape);
  Tensor invstd_expanded(invstd);

  batch_norm_backward_elemt_npu_expand_tensor(invstd_expanded, dim_c, input_ndim, input_shape);
  Tensor weight_expanded(weight);

  batch_norm_backward_elemt_npu_expand_tensor(weight_expanded, dim_c, input_ndim, input_shape);
  Tensor mean_dy_expanded(mean_dy);

  batch_norm_backward_elemt_npu_expand_tensor(mean_dy_expanded, dim_c, input_ndim, input_shape);
  Tensor mean_dy_xmu_expanded(mean_dy_xmu);

  batch_norm_backward_elemt_npu_expand_tensor(mean_dy_xmu_expanded, dim_c, input_ndim, input_shape);
  Tensor grad_input = OpPreparation::ApplyTensor(input);
  return batch_norm_backward_elemt_npu_nocheck(
      grad_out,
      input,
      mean_expanded,
      invstd_expanded,
      weight_expanded,
      mean_dy_expanded,
      mean_dy_xmu_expanded,
      grad_input);
}

} // namespace native
} // namespace at
