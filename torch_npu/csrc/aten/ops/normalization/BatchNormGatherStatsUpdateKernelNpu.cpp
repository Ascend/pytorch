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

std::tuple<at::Tensor &, at::Tensor &> batch_norm_gather_stats_update_npu_impl(
    at::Tensor &mean_all,
    at::Tensor &invstd_all,
    const at::Tensor &self,
    const at::Tensor &sum,
    const at::Tensor &square_sum,
    const at::Tensor &running_mean,
    const at::Tensor &running_var,
    double momentum,
    double eps,
    const at::Tensor &counts) {
  at::Tensor counts_cp = counts.scalar_type() == at::kInt ? counts : NPUNativeFunctions::npu_dtype_cast(counts, at::kInt);

  auto running_mean_dtype = running_mean.scalar_type();
  at::Tensor running_mean_ = NPUNativeFunctions::npu_dtype_cast(NPUNativeFunctions::npu_format_cast((running_mean.defined() ? 
      running_mean : at::native::zeros({self.size(1)}, sum.options())), ACL_FORMAT_ND), sum.scalar_type());
  at::Tensor running_var_ = NPUNativeFunctions::npu_dtype_cast(NPUNativeFunctions::npu_format_cast((running_var.defined() ? 
      running_var : at::native::ones({self.size(1)}, sum.options())), ACL_FORMAT_ND), sum.scalar_type());

  OpCommand cmd;
  cmd.Name("SyncBatchNormGatherStats")
     .Input(sum)
     .Input(square_sum)
     .Input(counts_cp)
     .Input(running_mean_)
     .Input(running_var_)
     .Output(mean_all)
     .Output(invstd_all)
     .Output(running_mean_)
     .Output(running_var_)
     .Attr("momentum", static_cast<float>(momentum))
     .Attr("eps", static_cast<float>(eps))
     .Run();

  if (running_mean.defined()) {
    if (running_mean_.scalar_type() != running_mean_dtype) {
      running_mean_ = NPUNativeFunctions::npu_dtype_cast(running_mean_, running_mean_dtype);
      running_var_ = NPUNativeFunctions::npu_dtype_cast(running_var_, running_mean_dtype);
    }
    running_mean.copy_(running_mean_);
    running_var.copy_(running_var_);
  }
  return std::tie(mean_all, invstd_all);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::batch_norm_gather_stats_update(
    const at::Tensor &self,
    const at::Tensor &sum,
    const at::Tensor &square_sum,
    const c10::optional<at::Tensor> &running_mean_opt,
    const c10::optional<at::Tensor> &running_var_opt,
    double momentum,
    double eps,
    const at::Tensor &counts) {
  c10::SmallVector<int64_t, N> output_size = {self.size(1)};

  const at::Tensor &running_mean = c10::value_or_else(running_mean_opt, 
                                                      []{ return at::Tensor(); });
  const at::Tensor &running_var = c10::value_or_else(running_var_opt, 
                                                     []{ return at::Tensor(); });

  at::Tensor mean_all = OpPreparation::ApplyTensor(sum, output_size);
  at::Tensor invstd_all = OpPreparation::ApplyTensor(sum, output_size);

  batch_norm_gather_stats_update_npu_impl(mean_all, invstd_all, self, sum, square_sum,
                                          running_mean, running_var, momentum, eps, counts);

  return std::make_tuple(mean_all, invstd_all);
}

} // namespace native
} // namespace at_npu
