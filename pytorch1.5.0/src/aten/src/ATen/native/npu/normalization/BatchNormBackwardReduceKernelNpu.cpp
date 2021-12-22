
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

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> batch_norm_backward_reduce_npu_impl(
    Tensor& sum_dy,
    Tensor& sum_dy_xmu,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& grad_out,
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const Tensor& weight,
    bool input_g,
    bool weight_g,
    bool bias_g,
    bool isFullyFp16 = false) {
  Tensor sum_dy_;
  Tensor sum_dy_xmu_;
  Tensor grad_bias_;

  Tensor grad_out_ = grad_out.scalar_type() == at::kFloat ? grad_out : grad_out.npu_dtype_cast(at::kFloat);
  Tensor self_ = self.scalar_type() == at::kFloat ? self : self.npu_dtype_cast(at::kFloat);
  Tensor mean_ = mean.scalar_type() == at::kFloat ? mean : mean.npu_dtype_cast(at::kFloat);
  Tensor invstd_ = invstd.scalar_type() == at::kFloat ? invstd : invstd.npu_dtype_cast(at::kFloat);
  Tensor weight_ = weight.scalar_type() == at::kFloat ? weight : weight.npu_dtype_cast(at::kFloat);

  SmallVector<int64_t, N> axes;
  int dimN = self_.ndimension();
  for(int i = 0; i < dimN; i++){
    if (i == 1) {
      continue;
    }
    axes.emplace_back(i);
  }
  // sum_dy_xmu
  Tensor mul_dy_dx = grad_out_ * self_;
  sum_dy_xmu_ = at::sum(mul_dy_dx, axes, false);
  grad_bias_ = at::sum(grad_out_, axes, false);
  sum_dy_ = grad_bias_;

  // grad_weight
  Tensor sum_dy_xmu_out = OpPreparation::ApplyTensor(sum_dy_);
  Tensor grad_weight_res = OpPreparation::ApplyTensor(invstd_);
  OpCommand cmd;
  cmd.Name("SyncBatchNormBackwardReduce")
      .Input(sum_dy_)
      .Input(sum_dy_xmu_)
      .Input(mean_)
      .Input(invstd_)
      .Output(sum_dy_xmu_out)
      .Output(grad_weight_res)
      .Run();
  if (input_g){
    sum_dy_xmu.copy_(sum_dy_xmu_out);
    sum_dy.copy_(sum_dy_);
  }
  if (weight_g) {
    grad_weight.copy_(grad_weight_res);
  }
  if (bias_g) {
    grad_bias.copy_(grad_bias_);
  }

  return std::tie(sum_dy, sum_dy_xmu, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_npu(
    const Tensor& grad_out,
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const Tensor& weight,
    bool input_g,
    bool weight_g,
    bool bias_g) {
  TORCH_CHECK(
      self.scalar_type() == grad_out.scalar_type(),
      "Expected input's dtype equal grad_out's dtype ",
      grad_out.scalar_type(),
      "But found ",
      self.scalar_type());
  bool isFullyFp16 = false;
  if (self.scalar_type() == mean.scalar_type() && self.scalar_type() == at::kHalf) {
    isFullyFp16 = true;
  }
  int64_t n_input = self.size(1);
  Tensor sum_dy_;
  Tensor sum_dy_xmu_;
  Tensor grad_weight_;
  Tensor grad_bias_;

  Tensor weight_ = weight.defined() ? weight : ones_npu({n_input}, self.options().dtype(
      isFullyFp16 ? at::kHalf : at::kFloat));

  if (input_g) {
      sum_dy_ = OpPreparation::ApplyTensor(mean, mean.options().dtype(isFullyFp16 ? at::kHalf : at::kFloat));
      sum_dy_xmu_ = OpPreparation::ApplyTensor(mean, mean.options().dtype(isFullyFp16 ? at::kHalf : at::kFloat));
  }
  if (weight_g) {
      grad_weight_ = OpPreparation::ApplyTensor({n_input}, weight_.options().dtype(
          isFullyFp16 ? at::kHalf : at::kFloat), weight_);
  }
  if (bias_g) {
      grad_bias_ = OpPreparation::ApplyTensor({n_input}, weight_.options().dtype(
          isFullyFp16 ? at::kHalf : at::kFloat), weight_);
  }
  batch_norm_backward_reduce_npu_impl(
      sum_dy_,
      sum_dy_xmu_,
      grad_weight_,
      grad_bias_,
      grad_out,
      self,
      mean,
      invstd,
      weight,
      input_g,
      weight_g,
      bias_g,
      isFullyFp16);
  return std::tie(sum_dy_, sum_dy_xmu_, grad_weight_, grad_bias_);
}

} // namespace native
} // namespace at
