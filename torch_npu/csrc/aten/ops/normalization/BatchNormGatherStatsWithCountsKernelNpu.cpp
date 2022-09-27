// Copyright (c) 2022 Huawei Technologies Co., Ltd
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

std::tuple<at::Tensor&, at::Tensor&> batch_norm_gather_stats_with_counts_npu_impl(
    at::Tensor& mean_all,
    at::Tensor& invstd_all,
    const at::Tensor& self,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    double momentum,
    double eps,
    const at::Tensor& counts) {
  auto options = self.options();
  auto dimC = self.size(1);
  at::Tensor meanCp = NPUNativeFunctions::npu_dtype_cast(mean, at::kFloat);
  at::Tensor invstdCp = NPUNativeFunctions::npu_dtype_cast(invstd, at::kFloat);
  auto running_mean_dtype = running_mean.scalar_type();
  at::Tensor running_mean_ = NPUNativeFunctions::npu_dtype_cast(NPUNativeFunctions::npu_format_cast((running_mean.defined() ?
      running_mean.unsqueeze(0) :at::zeros({1, dimC}, options)), ACL_FORMAT_ND), at::kFloat);
  at::Tensor running_var_ = NPUNativeFunctions::npu_dtype_cast(NPUNativeFunctions::npu_format_cast((running_var.defined() ?
      running_var.unsqueeze(0) :at::ones({1, dimC}, options)), ACL_FORMAT_ND), at::kFloat);
  at::IntArrayRef axes({0});
  at::Tensor countsTensor;
  countsTensor = NPUNativeFunctions::npu_dtype_cast(counts, meanCp.scalar_type());
  at::Tensor countsTensorT = countsTensor.unsqueeze(-1);
  at::Tensor countsTensorBroadcast = NPUNativeFunctions::npu_broadcast(countsTensorT, invstd.sizes());
  at::Tensor countsAllSum = OpPreparation::ApplyTensorWithSizes({1, dimC}, meanCp.options());
  OpCommand cmd1;
  cmd1.Name("ReduceSum")
      .Input(countsTensorBroadcast)
      .Input(axes, at::kInt)
      .Attr("keep_dims", true)
      .Output(countsAllSum)
      .Run();

  at::Tensor countsAllSumBroadcast = countsAllSum.expand(countsTensorBroadcast.sizes());
  OpCommand cmd2;
  cmd2.Name("ReduceMeanWithCount")
      .Input(meanCp)
      .Input(countsTensorBroadcast)
      .Input(countsAllSumBroadcast)
      .Output(mean_all)
      .Attr("axes", axes)
      .Attr("keep_dims", true)
      .Run();

  at::Tensor meanBroadcast = mean_all.expand(mean.sizes());
  OpCommand cmd3;
  cmd3.Name("SyncBatchNormGatherStatsWithCounts")
      .Input(meanCp)
      .Input(invstdCp)
      .Input(countsTensorBroadcast)
      .Input(meanBroadcast)
      .Input(countsAllSum)
      .Input(running_var_)
      .Output(invstd_all)
      .Output(running_var_)
      .Attr("momentum", static_cast<float>(momentum))
      .Attr("epsilon", static_cast<float>(eps))
      .Run();

  if (running_mean.defined()){
    OpCommand cmd4;
    cmd4.Name("SyncBNTrainingUpdate")
        .Input(mean_all)
        .Input(running_mean_)
        .Output(running_mean_)
        .Attr("momentum", static_cast<float>(momentum))
        .Run();
    // running_mean almost apply is the same as running_var
    if (running_mean_.scalar_type() != running_mean_dtype) {
      running_mean_ = NPUNativeFunctions::npu_dtype_cast(running_mean_, running_mean_dtype);
      running_var_ = NPUNativeFunctions::npu_dtype_cast(running_var_, running_mean_dtype);
    }
    running_mean.copy_(running_mean_.squeeze(0));
    running_var.copy_(running_var_.squeeze(0));
  }

  return std::tie(mean_all, invstd_all);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::batch_norm_gather_stats_with_counts(
    const at::Tensor& self,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    double momentum,
    double eps,
    const at::Tensor& counts) {
  const at::Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return at::Tensor();});
  const at::Tensor& running_var = c10::value_or_else(running_var_opt, [] {return at::Tensor();});
  bool isFullyFp16 = false;
  if (self.scalar_type() == mean.scalar_type() && self.scalar_type() == at::kHalf) {
    isFullyFp16 = true;
  }

  at::Tensor mean_all = OpPreparation::ApplyTensor({1, self.size(1)}, self.options().dtype(at::kFloat), self);
  at::Tensor invstd_all = OpPreparation::ApplyTensor({1, self.size(1)}, self.options().dtype(at::kFloat), self);

  batch_norm_gather_stats_with_counts_npu_impl(mean_all, invstd_all, self,
      mean, invstd, running_mean, running_var,
      momentum, eps, counts);

  if (isFullyFp16) {
    mean_all = NPUNativeFunctions::npu_dtype_cast(mean_all, at::kHalf);
    invstd_all = NPUNativeFunctions::npu_dtype_cast(invstd_all, at::kHalf);
  }
  return std::make_tuple(mean_all.squeeze(0), invstd_all.squeeze(0));
}

} // namespace native
} // namespace at_npu