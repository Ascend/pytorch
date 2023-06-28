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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

auto check_and_trans_dim(const at::Tensor& self, at::IntArrayRef dim) {
  int64_t dim_size = self.dim();
  int64_t ne_dim_size = dim_size * -1;
  std::vector<int64_t> result_dim;
  for(int64_t i = 0; i < dim.size(); i++) {
    if(dim[i] >= ne_dim_size && dim[i] <= (dim_size - 1)) {
      int64_t tmp_dim = CalcuOpUtil::MakeWrapDim(dim[i], self.dim());
      result_dim.emplace_back(tmp_dim);
    } else {
      AT_ERROR("dim value should be in the range of [-n, n-1], n is the dimension number of input tensor.");
    }
  }
  std::sort(result_dim.begin(), result_dim.end());
  return result_dim;
}

auto get_result_names(const at::Tensor& self, at::IntArrayRef dim, bool keepdim){
  auto names = self.names();
  std::vector<at::Dimname> result_names;
  for(int64_t i = 0; i < names.size(); i++){
    result_names.emplace_back(names[i]);
  }
  if(!keepdim){
    for(int64_t i = dim.size() - 1; i >= 0; i--){
      int64_t need_remove_dim = dim[i];
      result_names.erase(result_names.begin() + need_remove_dim);
    }
  }
  return result_names;
}

at::Tensor& var_after_npu_nocheckout(
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

int64_t get_shape_prod(const at::Tensor& self, at::IntArrayRef dim) {
  int64_t shape_prod = 1;
  if (self.dim() == 0) {
    shape_prod = 1;
  } else if (dim.size() == 0) {
    for (auto i = 0; i < self.dim(); i++) {
      shape_prod *= self.size(i);
    }
  } else {
    for(auto i = 0; i < dim.size(); i++) {
      shape_prod *= self.size(dim[i]);
    }
  }
  return shape_prod;
}

tuple<at::Tensor&, at::Tensor&> var_mean_compute(
    at::Tensor& variance,
    at::Tensor& mean,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  auto meanOutputSizeKeepDim = var_npu_output_size(self, dim, true);
  auto meanOutputSizeNotKeepDim = var_npu_output_size(self, dim, false);
  mean = at::mean(self, dim, false);
  NPUNativeFunctions::resize_(mean, meanOutputSizeKeepDim, c10::nullopt);
  at::Tensor mean_broadcast = NPUNativeFunctions::npu_broadcast(mean, self.sizes());
  if(!keepdim){
    NPUNativeFunctions::resize_(mean, meanOutputSizeNotKeepDim, c10::nullopt);
  }
  if (unbiased) {
    auto shape_prod = get_shape_prod(self, dim);
    if (shape_prod <= 1) {
      variance.fill_(NAN);
      return tuple<at::Tensor&, at::Tensor&>(variance, mean);
    }
  }
  var_after_npu_nocheckout(variance, self, mean_broadcast, dim, unbiased, keepdim);
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
  auto meanOutputSizeKeepDim = var_npu_output_size(self, dim_now, true);
  auto meanOutputSizeNotKeepDim = var_npu_output_size(self, dim_now, false);
  auto ori_type = self.scalar_type();
  if(ori_type != c10::ScalarType::Half && ori_type != c10::ScalarType::Float) {
    AT_ERROR("Var Mean only support float16 or float32 type.");
  }
  if(variance.scalar_type() != mean.scalar_type() || variance.scalar_type() != ori_type) {
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
    at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction,
    bool keepdim,
    at::Tensor& var) {
  auto unbiased = !(correction.has_value() && correction.value() == 0);
  // check and trans dim
  auto dim_now = check_and_trans_dim(self, dim.value_or(at::IntArrayRef{}));
  auto outputSize = var_npu_output_size(self, dim_now, keepdim);

  // construct the output mean tensor of the NPU
  at::Tensor mean = OpPreparation::ApplyTensor(self, outputSize);
  at::Tensor var_ = OpPreparation::ApplyTensor(self, outputSize);

  var_mean_out_npu(var_, mean, self, dim.value_or(at::IntArrayRef{}), unbiased, keepdim);
  OpPreparation::CheckOut(
      {var_},
      var,
      var_);
      var.copy_(var_);
   return var;
}

at::Tensor& NPUNativeFunctions::var_out(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& var) {
  return at::var_out(var, self, dimnames_to_positions(self, dim),
      c10::make_optional<int64_t>({unbiased ? 1 : 0}), keepdim);
}

at::Tensor& NPUNativeFunctions::var_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& var) {
  return at::var_out(var, self, dim,
      c10::make_optional<int64_t>({unbiased ? 1 : 0}), keepdim);
}

at::Tensor NPUNativeFunctions::var(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  auto unbiased = !(correction.has_value() && correction.value() == 0);
  auto dim_now = check_and_trans_dim(self, dim.value_or(at::IntArrayRef{}));
  // calculate the output size
  auto outputSize = var_npu_output_size(self, dim_now, keepdim);

  // construct the output tensor of the NPU
  at::Tensor variance = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  NPUNativeFunctions::var_out(self, dim.value_or(at::IntArrayRef{}), unbiased, keepdim, variance);

  return variance;
}

at::Tensor NPUNativeFunctions::var(const at::Tensor& self, bool unbiased) {
  bool keepdim = false;
  c10::SmallVector<int64_t, N> dim = CalcuOpUtil::GetDimlistForTensor(self);

  return at::var(self, dim, c10::make_optional<int64_t>({unbiased ? 1 : 0}), keepdim);
}

at::Tensor NPUNativeFunctions::var(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  return at::var(self, dim, c10::make_optional<int64_t>({unbiased ? 1 : 0}), keepdim);
}

at::Tensor NPUNativeFunctions::var(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim) {
  return at::var(self, dimnames_to_positions(self, dim),
      c10::make_optional<int64_t>({unbiased ? 1 : 0}), keepdim);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::var_mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  auto unbiased = !(correction.has_value() && correction.value() == 0);
  auto dim_now = check_and_trans_dim(self, dim.value_or(at::IntArrayRef{}));
  // calculate the output size
  auto outputSize = var_npu_output_size(self, dim_now, keepdim);

  // construct the output tensor of the NPU
  at::Tensor variance = OpPreparation::ApplyTensor(self, outputSize);

  at::Tensor mean = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  var_mean_out_npu(variance, mean, self, dim.value_or(at::IntArrayRef{}), unbiased, keepdim);

  return tuple<at::Tensor, at::Tensor>(variance, mean);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::var_mean(const at::Tensor& self, bool unbiased) {
  c10::SmallVector<int64_t, SIZE> dim = CalcuOpUtil::GetDimlistForTensor(self);

  return at::var_mean(self, dim, c10::make_optional<int64_t>({unbiased ? 1 : 0}), false);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::var_mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  return at::var_mean(self, dim, c10::make_optional<int64_t>({unbiased ? 1 : 0}), keepdim);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::var_mean(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim) {
  return at::var_mean(self, dimnames_to_positions(self, dim),
      c10::make_optional<int64_t>({unbiased ? 1 : 0}), keepdim);
}

} // namespace native
} // namespace at_npu
