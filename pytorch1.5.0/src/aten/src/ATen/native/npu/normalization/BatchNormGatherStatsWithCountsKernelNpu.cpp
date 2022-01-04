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

std::tuple<Tensor&, Tensor&> batch_norm_gather_stats_with_counts_npu_impl(
    Tensor& mean_all,
    Tensor& invstd_all,
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum,
    double eps,
    IntArrayRef counts) {
  auto options = self.options();
  auto dimC = self.size(1);

  Tensor meanCp = mean.npu_dtype_cast(at::kFloat);
  Tensor invstdCp = invstd.npu_dtype_cast(at::kFloat);

  auto running_mean_dtype = running_mean.scalar_type();

  Tensor running_mean_ = (running_mean.defined() ? running_mean.unsqueeze(0) :
      zeros_npu({1, dimC}, options)).npu_format_cast(ACL_FORMAT_ND).npu_dtype_cast(at::kFloat);
  Tensor running_var_ = (running_var.defined() ? running_var.unsqueeze(0) :
      ones_npu({1, dimC}, options)).npu_format_cast(ACL_FORMAT_ND).npu_dtype_cast(at::kFloat);
  IntArrayRef axes({0});
  Tensor countsTensor;
  // create countsTensor
  {
    SmallVector<int64_t, N> countList = array_to_small_vector(counts);
    auto cpuTensor = at::empty(countList.size(), TensorOptions(kCPU).dtype(at::kLong));
    std::memcpy(cpuTensor.data_ptr(), (void*)countList.data(), sizeof(int64_t) * cpuTensor.numel());
    countsTensor = cpuTensor.to(at::kNPU).npu_dtype_cast(meanCp.scalar_type());
  }
  Tensor countsTensorT = transpose_npu(countsTensor.unsqueeze(-1), {0, 1});
  Tensor countsTensorBroadcast = npu_broadcast(countsTensorT, invstd.sizes());

  Tensor countsAllSum = OpPreparation::ApplyTensorWithSizes({1, dimC}, meanCp.options());
  OpCommand cmd1;
  cmd1.Name("ReduceSum")
      .Input(countsTensorBroadcast)
      .Input(axes, at::kInt)
      .Attr("keep_dims", true)
      .Output(countsAllSum)
      .Run();

  Tensor countsAllSumBroadcast = countsAllSum.expand(countsTensorBroadcast.sizes());
  OpCommand cmd2;
  cmd2.Name("ReduceMeanWithCount")
      .Input(meanCp)
      .Input(countsTensorBroadcast)
      .Input(countsAllSumBroadcast)
      .Output(mean_all)
      .Attr("axes", axes)
      .Attr("keep_dims", true)
      .Run();

  Tensor meanBroadcast = mean_all.expand(mean.sizes());
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
      running_mean_ = running_mean_.npu_dtype_cast(running_mean_dtype);
      running_var_ = running_var_.npu_dtype_cast(running_mean_dtype);
    }
    running_mean.copy_(running_mean_.squeeze(0));
    running_var.copy_(running_var_.squeeze(0));
  }

  return std::tie(mean_all, invstd_all);
}

std::tuple<Tensor, Tensor> batch_norm_gather_stats_with_counts_npu(
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum,
    double eps,
    IntArrayRef counts) {
  bool isFullyFp16 = false;
  if (self.scalar_type() == mean.scalar_type() && self.scalar_type() == at::kHalf) {
    isFullyFp16 = true;
  }

  Tensor mean_all = OpPreparation::ApplyTensor({1, self.size(1)}, self.options().dtype(at::kFloat), self);
  Tensor invstd_all = OpPreparation::ApplyTensor({1, self.size(1)}, self.options().dtype(at::kFloat), self);
  
  batch_norm_gather_stats_with_counts_npu_impl(mean_all, invstd_all, self,
      mean, invstd, running_mean, running_var,
      momentum, eps, counts);
      
  if (isFullyFp16) {
    mean_all = mean_all.npu_dtype_cast(at::kHalf);
    invstd_all = invstd_all.npu_dtype_cast(at::kHalf);
  }
  return std::make_tuple(mean_all.squeeze(0), invstd_all.squeeze(0));
}

} // namespace native
} // namespace at
