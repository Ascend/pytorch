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

#include <c10/npu/NPUCachingAllocator.h>
#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

namespace{
Tensor& batch_norm_infer_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
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

tuple<Tensor&, Tensor&> batch_norm_training_reduce_nocheck(
    Tensor& sum,
    Tensor& square_sum,
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  OpCommand cmd;
  if (self.dim() == 5) {
    // Used for 3D BatchNorm in Training
    cmd.Name("BN3DTrainingReduce")
        .Input(self, "x", ACL_FORMAT_NCDHW)
        .Output(sum, "sum", ACL_FORMAT_NCHW)
        .Output(square_sum, "square_sum", ACL_FORMAT_NCHW)
        .Attr("epsilon", static_cast<float>(eps))
        .Run();
  } else {
    // Used for 2D BatchNorm in Training
    cmd.Name("BNTrainingReduce")
        .Input(self, "x", ACL_FORMAT_NCHW)
        .Output(sum, "sum", ACL_FORMAT_NCHW)
        .Output(square_sum, "square_sum", ACL_FORMAT_NCHW)
        .Attr("epsilon", static_cast<float>(eps))
        .Run();
  }

  return tuple<Tensor&, Tensor&>(sum, square_sum);
}

tuple<Tensor&, Tensor&, Tensor&> batch_norm_training_update_nocheck(
    Tensor& result,
    Tensor& save_mean,
    Tensor& save_invstd,
    const Tensor& self,
    const Tensor& sum,
    const Tensor& square_sum,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  OpCommand cmd;
  if (self.dim() == 5) {
    // Used for 3D BatchNorm in Training
    cmd.Name("BN3DTrainingUpdate")
        .Input(self, "x", ACL_FORMAT_NCDHW)
        .Input(sum, "sum", ACL_FORMAT_NCHW)
        .Input(square_sum, "square_sum", ACL_FORMAT_NCHW)
        .Input(weight, "scale", ACL_FORMAT_NCHW)
        .Input(bias, "offset", ACL_FORMAT_NCHW)
        .Input(running_mean, "mean", ACL_FORMAT_NCHW)
        .Input(running_var, "variance", ACL_FORMAT_NCHW)
        .Output(result, "y", ACL_FORMAT_NCDHW)
        .Output(const_cast<Tensor&>(running_mean), "mean", ACL_FORMAT_NCHW)
        .Output(const_cast<Tensor&>(running_var), "variance", ACL_FORMAT_NCHW)
        .Output(save_mean, "batch_mean", ACL_FORMAT_NCHW)
        .Output(save_invstd, "batch_variance", ACL_FORMAT_NCHW)
        .Attr("epsilon", static_cast<float>(eps))
        .Attr("factor", static_cast<float>(momentum))
        .Run();
  } else {
    // Used for 2D BatchNorm in Training
    cmd.Name("BNTrainingUpdate")
        .Input(self, "x", ACL_FORMAT_NCHW)
        .Input(sum, "sum", ACL_FORMAT_NCHW)
        .Input(square_sum, "square_sum", ACL_FORMAT_NCHW)
        .Input(weight, "scale", ACL_FORMAT_NCHW)
        .Input(bias, "offset", ACL_FORMAT_NCHW)
        .Input(running_mean, "mean", ACL_FORMAT_NCHW)
        .Input(running_var, "variance", ACL_FORMAT_NCHW)
        .Output(result, "y", ACL_FORMAT_NCHW)
        .Output(const_cast<Tensor&>(running_mean), "mean", ACL_FORMAT_NCHW)
        .Output(const_cast<Tensor&>(running_var), "variance", ACL_FORMAT_NCHW)
        .Output(save_mean, "batch_mean", ACL_FORMAT_NCHW)
        .Output(save_invstd, "batch_variance", ACL_FORMAT_NCHW)
        .Attr("epsilon", static_cast<float>(eps))
        .Attr("factor", static_cast<float>(momentum))
        .Run();
  }

  return tuple<Tensor&, Tensor&, Tensor&>(result, save_mean, save_invstd);
}

tuple<Tensor&, Tensor&, Tensor&> batch_norm_impl(
    Tensor& result,
    Tensor& save_mean,
    Tensor& save_invstd,
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
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
    return tuple<Tensor&, Tensor&, Tensor&>(result, save_mean, save_invstd);
  }

  // calculate the output result of the NPU
  Tensor sum = OpPreparation::ApplyTensor(running_mean.sizes(), running_mean.options().dtype(at::kFloat), running_mean);
  Tensor square_sum = OpPreparation::ApplyTensor(running_mean.sizes(), running_mean.options().dtype(at::kFloat), running_mean);

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

  return tuple<Tensor&, Tensor&, Tensor&>(result, save_mean, save_invstd);
}
} // namespace

tuple<Tensor, Tensor, Tensor> batch_norm_npu(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  Tensor self_reshape;
  SmallVector<int64_t, N> self_shape = array_to_small_vector(self.sizes());

  int64_t self_npu_format = CalcuOpUtil::get_tensor_npu_format(self);
  // BatchNorm is axis sensitive, the size of mean/var depends on dim_c.
  if (self_npu_format == ACL_FORMAT_NDHWC ||
      self_npu_format == ACL_FORMAT_NHWC) {
    AT_ERROR(
        "Tensor with channel last format (",
        self_npu_format,
        ") is not supported in BatchNorm.");
  }

  if (self.dim() <= 4) {
    SmallVector<int64_t, N> nchw_shape(self_shape);
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
    SmallVector<int64_t, N> nchw_shape = {self_shape[0] * self_shape[2], self_shape[1], self_shape[3], self_shape[4]};
    // ndchw -> nchw
    self_reshape = self_reshape.reshape(nchw_shape);
  }

  // process when affine=Flase and track_running_stats=False
  int64_t dim_c = self_reshape.size(1);
  TensorOptions options = self.options().dtype(ScalarType::Float);

  // 2D/3D BN Ops support ACL_FORMAT_NC1HWC0 format tensor(1D).
  Tensor running_mean_tensor = running_mean.defined() ? running_mean.npu_format_cast_(ACL_FORMAT_NC1HWC0) : zeros_npu({dim_c}, options);
  Tensor running_var_tensor = running_var.defined() ? running_var.npu_format_cast_(ACL_FORMAT_NC1HWC0) : ones_npu({dim_c}, options);

  Tensor weight_tensor = weight.defined() ? weight.npu_format_cast_(ACL_FORMAT_NC1HWC0) : ones_npu({dim_c}, options);
  Tensor bias_tensor = bias.defined() ? bias.npu_format_cast_(ACL_FORMAT_NC1HWC0) : zeros_npu({dim_c}, options);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self_reshape.sizes(), self_reshape.options(), self_reshape);

  Tensor save_mean;
  Tensor save_invstd;
  if (train) {
    save_mean = OpPreparation::ApplyTensor(running_mean_tensor.sizes(), running_mean_tensor.options().dtype(at::kFloat), running_mean_tensor);
    save_invstd = OpPreparation::ApplyTensor(running_var_tensor.sizes(), running_var_tensor.options().dtype(at::kFloat), running_var_tensor);
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
    swap(self_shape[1], self_shape[2]);
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
} // namespace at
