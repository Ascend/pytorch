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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& batch_norm_elemt_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double eps) {
  auto dimC = self.size(1);
  auto options = self.options().dtype(at::kFloat);
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});

  at::Tensor weight_ = weight.defined() ? weight : at::ones({dimC}, options);
  at::Tensor bias_ = bias.defined() ? bias : at::ones({dimC}, options);
  at::Tensor mean_ = mean.defined() ? mean : at::ones({dimC}, options);
  at::Tensor invstd_ = invstd.defined() ? invstd : at::ones({dimC}, options);
  TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 && mean.dim() == 1 && invstd.dim() == 1,
              "weight, bias, mean, invstd: must be only one dimension.");
  TORCH_CHECK(weight.size(0) == dimC && bias.size(0) == dimC && mean.size(0) == dimC && invstd.size(0) == dimC,
              "weight, bias, mean, invstd: shape must be equal to  self's dimC.");
  at::Tensor one = at::ones({1}, options);
  auto variance = at::mul(invstd_, invstd_);
  variance = at::div(one, variance) - eps;
  int64_t selfDim = self.dim();
  at::Tensor self5d(self);
  c10::SmallVector<int64_t, N> selfShape = array_to_small_vector(self.sizes());
  if (selfDim > 5) {
    self5d = self.reshape({self.size(0), self.size(1), self.size(2), self.size(3), -1});
  }
  std::tuple<at::Tensor, at::Tensor, at::Tensor> bnReult = at::native_batch_norm(
      self5d, weight_, bias_, mean_, variance, false, 0.0, eps);
  result.copy_(std::get<0>(bnReult));
  if (selfDim > 5) {
    result = result.view(selfShape);
  }
  return result;
}

at::Tensor& NPUNativeFunctions::batch_norm_elemt_out(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double eps,
    at::Tensor& result) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  OpPreparation::CheckOut({self}, result, self);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self, weight, bias, mean, invstd}, {result})
        .Func([&self, &weight, &bias, &mean, &invstd, &eps](at::Tensor& result)
        {batch_norm_elemt_nocheck(result, self, weight, bias, mean, invstd, eps);})
        .Call(result);
}

at::Tensor NPUNativeFunctions::batch_norm_elemt(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double eps) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  batch_norm_elemt_nocheck(result, self, weight, bias, mean, invstd, eps);
  return result;
}

} // namespace native
} // namespace at