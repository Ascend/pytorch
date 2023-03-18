// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include <ATen/WrapDimUtils.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

auto check_and_trans_dim(const at::Tensor& self, at::IntArrayRef dim) {
  std::vector<int64_t> result_dim;
  auto self_dim = self.dim();
  for (int64_t i = 0; i < dim.size(); i++) {
      int64_t tmp_dim = c10::maybe_wrap_dim(dim[i], self_dim);
      result_dim.emplace_back(tmp_dim);
  }
  std::sort(result_dim.begin(), result_dim.end());
  return result_dim;
}

auto get_result_names(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
  auto names = self.names();
  std::vector<at::Dimname> result_names;
  for (int64_t i = 0; i < names.size(); i++) {
    result_names.emplace_back(names[i]);
  }

  if (!keepdim) {
    for (int64_t i = dim.size() - 1; i >= 0; i--) {
      int64_t need_remove_dim = dim[i];
      result_names.erase(result_names.begin() + need_remove_dim);
    }
  }
  return result_names;
}

at::Tensor& var_after_npu_nocheck(
    at::Tensor& var,
    const at::Tensor& self,
    const at::Tensor& mean_broadcast,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  bool if_std = false;
  OpCommand cmd;
  cmd.Name("ReduceStdV2Update")
     .Input(self)
     .Input(mean_broadcast)
     .Output(var)
     .Attr("dim", dim)
     .Attr("if_std", if_std)
     .Attr("unbiased", unbiased)
     .Attr("keepdim", keepdim)
     .Run();
  return var;
}

tuple<at::Tensor&, at::Tensor&> var_mean_compute(
    at::Tensor& variance,
    at::Tensor& mean,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  auto mean_output_size_keepdim = var_npu_output_size(self, dim, true);
  auto mean_output_size_not_keepdim = var_npu_output_size(self, dim, false);
  mean = at::mean(self, dim, false);
  mean.resize_(mean_output_size_keepdim);
  at::Tensor mean_broadcast = NPUNativeFunctions::npu_broadcast(mean, self.sizes());
  if (!keepdim) {
    mean.resize_(mean_output_size_not_keepdim);
  }
  var_after_npu_nocheck(variance, self, mean_broadcast, dim, unbiased, keepdim);
  return tuple<at::Tensor&, at::Tensor&>(variance, mean);
}

tuple<at::Tensor&, at::Tensor&> var_mean_out_npu(
    at::Tensor& variance,
    at::Tensor& mean,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim) {

  auto dim_now = check_and_trans_dim(self, dim);
  auto mean_output_size_keepdim = var_npu_output_size(self, dim_now, true);
  auto mean_output_size_not_keepdim = var_npu_output_size(self, dim_now, false);
  auto ori_type = self.scalar_type();

  if (ori_type != c10::ScalarType::Half && ori_type != c10::ScalarType::Float) {
    AT_ERROR("Var Mean only support float16 or float32 type.");
  }

  if (variance.scalar_type() != mean.scalar_type() || variance.scalar_type() != ori_type) {
    AT_ERROR("mean's type and variance' type must be equal to input's type.");
  }

  var_mean_compute(
      variance,
      mean,
      self,
      dim_now,
      unbiased,
      keepdim);

  return tuple<at::Tensor&, at::Tensor&>(variance, mean);
}

at::Tensor& NPUNativeFunctions::var_out(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& var) {
  return NPUNativeFunctions::var_out(
      self, dimnames_to_positions(self, dim), int64_t{unbiased ? 1 : 0}, keepdim, var);
}

at::Tensor& NPUNativeFunctions::var_out(
    const at::Tensor& self,
    at::DimnameList dim,
    c10::optional<int64_t> correction,
    bool keepdim,
    at::Tensor& var) {
  return NPUNativeFunctions::var_out(
      self, dimnames_to_positions(self, dim), correction, keepdim, var);
}

at::Tensor& NPUNativeFunctions::var_out(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& var) {
  return NPUNativeFunctions::var_out(self, c10::optional<at::IntArrayRef>(dim), int64_t{unbiased ? 1 : 0}, keepdim, var);
}

at::Tensor& NPUNativeFunctions::var_out(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> dims,
    c10::optional<int64_t> correction,
    bool keepdim,
    at::Tensor& var) {
  c10::SmallVector<int64_t, N> dim = CalcuOpUtil::GetDimlistForTensor(self);
  if (dims.has_value()) {
    dim = array_to_small_vector(dims.value());
  }

  auto dim_now = check_and_trans_dim(self, dim);
  auto output_size = var_npu_output_size(self, dim_now, keepdim);
  at::Tensor mean = OpPreparation::ApplyTensor(self, output_size);

  OpPreparation::CheckOut(
      {self},
      var,
      self,
      output_size);

  if (correction.has_value() && correction.value() == 0) {
    var_mean_out_npu(var, mean, self, dim, false, keepdim);
  } else {
    var_mean_out_npu(var, mean, self, dim, true, keepdim);
  }
  return var;
}

at::Tensor NPUNativeFunctions::var(
    const at::Tensor& self,
    at::DimnameList dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  return NPUNativeFunctions::var(self, dimnames_to_positions(self, dim), correction, keepdim);
}

at::Tensor NPUNativeFunctions::var(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> dims,
    c10::optional<int64_t> correction,
    bool keepdim) {
  c10::SmallVector<int64_t, N> dim = CalcuOpUtil::GetDimlistForTensor(self);
  if (dims.has_value()) {
    dim = array_to_small_vector(dims.value());
  }
  auto dim_now = check_and_trans_dim(self, dim);

  auto output_size = var_npu_output_size(self, dim_now, keepdim);
  at::Tensor variance = OpPreparation::ApplyTensor(self, output_size);
  NPUNativeFunctions::var_out(self, dims, correction, keepdim, variance);
  return variance;
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::var_mean(
    const at::Tensor& self,
    at::DimnameList dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  return NPUNativeFunctions::var_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::var_mean(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> dims,
    c10::optional<int64_t> correction,
    bool keepdim) {
  c10::SmallVector<int64_t, N> dim = CalcuOpUtil::GetDimlistForTensor(self);
  if (dims.has_value()) {
    dim = array_to_small_vector(dims.value());
  }
  auto dim_now = check_and_trans_dim(self, dim);
  auto output_size = var_npu_output_size(self, dim_now, keepdim);
  at::Tensor variance = OpPreparation::ApplyTensor(self, output_size);
  at::Tensor mean = OpPreparation::ApplyTensor(self, output_size);

  if (correction.has_value() && correction.value() == 0) {
    var_mean_out_npu(variance, mean, self, dim, false, keepdim);
  } else {
    var_mean_out_npu(variance, mean, self, dim, true, keepdim);
  }

  return tuple<at::Tensor, at::Tensor>(variance, mean);
}

} // namespace native
} // namespace at_npu
