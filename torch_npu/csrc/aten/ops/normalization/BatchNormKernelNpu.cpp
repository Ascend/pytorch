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
      .Input(self, "x", ACL_FORMAT_NCHW)
      .Input(weight, "scale", ACL_FORMAT_NCHW)
      .Input(bias, "offset", ACL_FORMAT_NCHW)
      .Input(running_mean, "mean", ACL_FORMAT_NCHW)
      .Input(running_var, "variance", ACL_FORMAT_NCHW)
      .Output(result, "y", ACL_FORMAT_NCHW)
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
  string name = (self.dim() == 5) ? "BN3DTrainingReduce" : "BNTrainingReduce";
  auto format = (self.dim() == 5) ? ACL_FORMAT_NCDHW : ACL_FORMAT_NCHW;
  
  cmd.Name(name)
      .Input(self, "x", format)
      .Output(sum, "sum", format)
      .Output(square_sum, "square_sum", format)
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
  string name = (self.dim() == 5) ? "BN3DTrainingUpdate" : "BNTrainingUpdate";
  auto format = (self.dim() == 5) ? ACL_FORMAT_NCDHW : ACL_FORMAT_NCHW;
  
  cmd.Name(name)
      .Input(self, "x", format)
      .Input(sum, "sum", format)
      .Input(square_sum, "square_sum", format)
      .Input(weight, "scale", format)
      .Input(bias, "offset", format)
      .Input(running_mean, "mean", format)
      .Input(running_var, "variance", format)
      .Output(result, "y", format)
      .Output(const_cast<at::Tensor&>(running_mean), "mean", format)
      .Output(const_cast<at::Tensor&>(running_var), "variance", format)
      .Output(save_mean, "batch_mean", format)
      .Output(save_invstd, "batch_variance", format)
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
  at::Tensor sum = OpPreparation::ApplyTensor(running_mean.sizes(), running_mean.options().dtype(at::kFloat), self);
  at::Tensor square_sum = OpPreparation::ApplyTensor(running_mean.sizes(), running_mean.options().dtype(at::kFloat), self);

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

  // BNTrainingUpdate can only support FP32 for mean and var
  auto running_mean_fp32 = running_mean;
  auto running_var_fp32 = running_var;
  auto weight_fp32 = weight;

  if (train && (running_mean.scalar_type() != at::kFloat)) {
    running_mean_fp32 = NPUNativeFunctions::npu_dtype_cast(running_mean, at::kFloat);
  }

  if (train && (running_var.scalar_type() != at::kFloat)) {
    running_var_fp32 = NPUNativeFunctions::npu_dtype_cast(running_var, at::kFloat);
  }

  // (Ascend) change dtype for matching op requirement.
  if (train && (weight.scalar_type() != at::kFloat)) {
    weight_fp32 = NPUNativeFunctions::npu_dtype_cast(weight, at::kFloat);
  }

  batch_norm_training_update_nocheck(
      result,
      save_mean,
      save_invstd,
      self,
      sum,
      square_sum,
      weight_fp32,
      bias,
      running_mean_fp32,
      running_var_fp32,
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

  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  const at::Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return at::Tensor();});
  const at::Tensor& running_var = c10::value_or_else(running_var_opt, [] {return at::Tensor();});

  at::Tensor self_reshape;
  c10::SmallVector<int64_t, N> self_shape = array_to_small_vector(self.sizes());

  int64_t self_npu_format = CalcuOpUtil::get_tensor_npu_format(self);
  // BatchNorm is axis sensitive, the size of mean/var depends on dim_c.
  if (self_npu_format == ACL_FORMAT_NDHWC ||
      self_npu_format == ACL_FORMAT_NHWC) {
    AT_ERROR(
        "at::Tensor with channel last format (",
        self_npu_format,
        ") is not supported in BatchNorm.");
  }

  if (self.dim() <= 4) {
    c10::SmallVector<int64_t, N> nchw_shape(self_shape);
    nchw_shape.resize(4, 1);
    self_reshape = self.reshape(nchw_shape);
  } else if (train && self.dim() == 5) {
    // Use 3D BN ops for training, merging axes is not required.
    self_reshape = self;
  } else {
    // Infering uses 2dInfer Op, case no matched 3DInfer Op
    // ncdhw -> ndchw
    self_reshape = self.permute({0, 2, 1, 3, 4});
    // nchw=(n*d, c, h, w)
    c10::SmallVector<int64_t, N> nchw_shape = {self_shape[0] * self_shape[2], self_shape[1], self_shape[3], self_shape[4]};
    // ndchw -> nchw
    self_reshape = self_reshape.reshape(nchw_shape);
  }

  // process when affine=Flase and track_running_stats=False
  int64_t dim_c = self_reshape.size(1);
  at::TensorOptions options = self.options().dtype(at::ScalarType::Float);

  at::Tensor weight_cp = weight;
  at::Tensor bias_cp = bias;
  at::Tensor running_mean_cp = running_mean;
  at::Tensor running_var_cp = running_var;

  // 2D/3D BN Ops support ACL_FORMAT_NC1HWC0 format tensor(1D).
  at::Tensor running_mean_tensor = running_mean.defined() ? running_mean_cp : at::zeros({dim_c}, options);
  at::Tensor running_var_tensor = running_var.defined() ? running_var_cp : at::ones({dim_c}, options);
  at::Tensor weight_tensor = weight.defined() ? weight_cp : at::ones({dim_c}, options);
  at::Tensor bias_tensor = bias.defined() ? bias_cp : at::zeros({dim_c}, options);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self_reshape.sizes(), self_reshape.options(), self_reshape);

  at::Tensor save_mean;
  at::Tensor save_invstd;
  if (train) {
    save_mean = OpPreparation::ApplyTensor(running_mean_tensor.sizes(), running_mean_tensor.options().dtype(at::kFloat), self);
    save_invstd = OpPreparation::ApplyTensor(running_var_tensor.sizes(), running_var_tensor.options().dtype(at::kFloat), self);
  } else {
    save_mean = {};
    save_invstd = {};
  }

  // calculate the output result of the NPU
  batch_norm_impl(
      result,
      save_mean,
      save_invstd,
      self_reshape,
      weight_tensor,
      bias_tensor,
      running_mean_tensor,
      running_var_tensor,
      train,
      momentum,
      eps);

  // Inverse reshape procedure using for recovering original shape of self.
  if (!train && self.dim() == 5) {
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
