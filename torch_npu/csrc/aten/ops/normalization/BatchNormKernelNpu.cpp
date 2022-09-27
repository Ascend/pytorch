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
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"


namespace at_npu {
namespace native {

namespace{
at::Tensor& batch_norm_infer_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  OpCommand cmd;
  cmd.Name("BNInfer")
      .Input(self)
      .Input(weight)
      .Input(bias)
      .Input(running_mean)
      .Input(running_var)
      .Output(result)
      .Attr("epsilon", static_cast<float>(eps))
      .Run();

  return result;
}

tuple<at::Tensor&, at::Tensor&> batch_norm_training_reduce_nocheck(
    at::Tensor& sum,
    at::Tensor& square_sum,
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  OpCommand cmd;
  cmd.Name("BNTrainingReduce")
      .Input(self)
      .Output(sum)
      .Output(square_sum)
      .Attr("epsilon", static_cast<float>(eps))
      .Run();

  return tuple<at::Tensor&, at::Tensor&>(sum, square_sum);
}

tuple<at::Tensor&, at::Tensor&, at::Tensor&> batch_norm_training_update_nocheck(
    at::Tensor& result,
    at::Tensor& save_mean,
    at::Tensor& save_invstd,
    const at::Tensor& self,
    const at::Tensor& sum,
    const at::Tensor& square_sum,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  OpCommand cmd;
  cmd.Name("BNTrainingUpdate")
      .Input(self)
      .Input(sum)
      .Input(square_sum)
      .Input(weight)
      .Input(bias)
      .Input(running_mean)
      .Input(running_var)
      .Output(result)
      .Output(const_cast<at::Tensor &>(running_mean))
      .Output(const_cast<at::Tensor &>(running_var))
      .Output(save_mean)
      .Output(save_invstd)
      .Attr("epsilon", static_cast<float>(eps))
      .Attr("factor", static_cast<float>(momentum))
      .Run();

  return tuple<at::Tensor&, at::Tensor&, at::Tensor&>(result, save_mean, save_invstd);
}

tuple<at::Tensor&, at::Tensor&, at::Tensor&> batch_norm_impl(
    at::Tensor& result,
    at::Tensor& save_mean,
    at::Tensor& save_invstd,
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  if (!train) {
    batch_norm_infer_nocheck(
        result,
        self,
        weight,
        bias,
        running_mean,
        running_var,
        train,
        momentum,
        eps);
    return tuple<at::Tensor&, at::Tensor&, at::Tensor&>(result, save_mean, save_invstd);
  }

  // calculate the output result of the NPU
  at::Tensor sum = OpPreparation::ApplyTensor(running_mean.sizes(), running_mean.options().dtype(at::kFloat), running_mean);
  at::Tensor square_sum = OpPreparation::ApplyTensor(running_mean.sizes(), running_mean.options().dtype(at::kFloat), running_mean);

  batch_norm_training_reduce_nocheck(
      sum,
      square_sum,
      self,
      weight,
      bias,
      running_mean,
      running_var,
      train,
      momentum,
      eps);

  batch_norm_training_update_nocheck(
      result,
      save_mean,
      save_invstd,
      self,
      sum,
      square_sum,
      weight,
      bias,
      running_mean,
      running_var,
      train,
      momentum,
      eps);

  return tuple<at::Tensor&, at::Tensor&, at::Tensor&>(result, save_mean, save_invstd);
}
} // namespace

tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::native_batch_norm(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps) {
  at::Tensor self_4d;
  c10::SmallVector<int64_t, N> self_shape = array_to_small_vector(self.sizes());

  if (self.dim() <= 4) {
    c10::SmallVector<int64_t, N> nchw_shape(self_shape);
    nchw_shape.resize(4, 1);
    self_4d = self.reshape(nchw_shape);
  } else if (self.dim() == 5) {
    // ncdhw -> ndchw
    self_4d = self.permute({0, 2, 1, 3, 4});
    // nchw=(n*d, c, h, w)
    c10::SmallVector<int64_t, N> nchw_shape = {self_shape[0] * self_shape[2], self_shape[1], self_shape[3], self_shape[4]};
    // ndchw -> nchw
    self_4d = self_4d.reshape(nchw_shape);
  }

  // process when affine=Flase and track_running_stats=False
  int64_t dim_c = self_4d.size(1);
  at::TensorOptions options = self.options().dtype(c10::ScalarType::Float);

  at::Tensor running_mean_tensor =
      (running_mean_opt.has_value() && running_mean_opt.value().defined()) ? running_mean_opt.value() : at::zeros({dim_c}, options);
  at::Tensor running_var_tensor =
      (running_var_opt.has_value() && running_var_opt.value().defined()) ? running_var_opt.value() : at::ones({dim_c}, options);

  at::Tensor weight_tensor =
      (weight_opt.has_value() && weight_opt.value().defined()) ? weight_opt.value() : at::ones({dim_c}, options);
  at::Tensor bias_tensor =
      (bias_opt.has_value() && bias_opt.value().defined()) ? bias_opt.value() : at::zeros({dim_c}, options);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self_4d.sizes(), self_4d.options(), self_4d);

  at::Tensor save_mean;
  at::Tensor save_invstd;
  if (train) {
    save_mean = OpPreparation::ApplyTensor(running_mean_tensor.sizes(), running_mean_tensor.options().dtype(at::kFloat), running_mean_tensor);
    save_invstd = OpPreparation::ApplyTensor(running_var_tensor.sizes(), running_var_tensor.options().dtype(at::kFloat), running_var_tensor);
  } else {
    //save_mean = {};
    //save_invstd = {};
  }

  // calculate the output result of the NPU
  batch_norm_impl(
      result,
      save_mean,
      save_invstd,
      self_4d,
      weight_tensor,
      bias_tensor,
      running_mean_tensor,
      running_var_tensor,
      train,
      momentum,
      eps);

  if (self.dim() == 5) {
    // NCHW -> NDCHW -> NCDHW
    std::swap(self_shape[1], self_shape[2]);
    result = result.view(self_shape);
    result = NpuUtils::format_contiguous(result);
    result = result.permute({0, 2, 1, 3, 4}).clone();
  } else if (self.dim() < 5) {
    result = result.view(self_shape);
    result = NpuUtils::format_contiguous(result);
  }

  return std::tie(result, save_mean, save_invstd);
}

} // namespace native
} // namespace at_npu
