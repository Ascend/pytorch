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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

int64_t calc_shape_prod(const at::Tensor& self, at::IntArrayRef dim) {
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

tuple<at::Tensor&, at::Tensor&> std_mean_out_npu_nocheck(
    at::Tensor& result_std,
    at::Tensor& result_mean,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim,
    int64_t correction) {
  OpCommand cmd1;
  cmd1.Name("ReduceMeanD")
      .Input(self)
      .Output(result_mean)
      .Attr("axes", dim)
      .Attr("keep_dims", keepdim)
      .Run();

  auto shape_prod = calc_shape_prod(self, dim);
  if (shape_prod == 0 || (shape_prod == 1 && shape_prod <= correction)) {
    result_std.fill_(NAN);
    return std::tie(result_std, result_mean);
  }
  if (correction > 1 && shape_prod <= correction) {
    result_std.fill_(INFINITY);
    return std::tie(result_std, result_mean);
  }

  at::Tensor result_mean_copy = result_mean;
  if (result_mean.dim() != 0 && keepdim == false) {
    auto dimVector = array_to_small_vector(dim);
    std::sort(dimVector.begin(), dimVector.end());
    for (int64_t i = 0; i < dimVector.size(); i++) {
      result_mean_copy = result_mean_copy.unsqueeze(dimVector[i]);
    }
  }
  result_mean_copy = result_mean_copy.expand(self.sizes());
  OpCommand cmd2;
  cmd2.Name("ReduceStdWithMean")
      .Input(self)
      .Input(result_mean_copy)
      .Output(result_std)
      .Attr("dim", dim)
      .Attr("unbiased", unbiased)
      .Attr("keepdim", keepdim)
      .Attr("correction", correction)
      .Run();

  return std::tie(result_std, result_mean);
}

at::Tensor& NPUNativeFunctions::std_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  return NPUNativeFunctions::std_out(self, dim, c10::make_optional<int64_t>({unbiased ? 1 : 0}), keepdim, result);
}

at::Tensor& NPUNativeFunctions::std_out(
    const at::Tensor& self,
    at::DimnameList dim,
    c10::optional<int64_t> correction,
    bool keepdim,
    at::Tensor& result) {
  return NPUNativeFunctions::std_out(self, dimnames_to_positions(self, dim), correction, keepdim, result);
}

at::Tensor& NPUNativeFunctions::std_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction,
    bool keepdim,
    at::Tensor& result) {
  c10::SmallVector<int64_t, SIZE> dims = CalcuOpUtil::GetDimlistForTensor(self);
  if (dim.has_value()) {
    dims = array_to_small_vector(dim.value());
  }
  auto output_size = reduce_ops_npu_output_size(self, dims, keepdim);
  at::Tensor mean_result = OpPreparation::ApplyTensor(self, output_size);
  auto real_correction = correction.has_value() ? correction.value() : 1;

  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      self.scalar_type(),
      output_size);

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    std_mean_out_npu_nocheck(contiguous_result, mean_result, self, dims, correction.has_value() ? true : false, keepdim, real_correction);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    std_mean_out_npu_nocheck(result, mean_result, self, dims, correction.has_value() ? true : false, keepdim, real_correction);
  }

  return result;
}

at::Tensor& NPUNativeFunctions::std_out(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  return NPUNativeFunctions::std_out(self, dimnames_to_positions(self, dim), unbiased, keepdim, result);
}

tuple <at::Tensor, at::Tensor> NPUNativeFunctions::std_mean(
    const at::Tensor & self,
    at::DimnameList dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  return NPUNativeFunctions::std_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}

tuple <at::Tensor, at::Tensor> NPUNativeFunctions::std_mean(
    const at::Tensor & self,
    at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  c10::SmallVector<int64_t, SIZE> dims = CalcuOpUtil::GetDimlistForTensor(self);
  if (dim.has_value()) {
    dims = array_to_small_vector(dim.value());
  }

  auto output_size = reduce_ops_npu_output_size(self, dims, keepdim);

  at::Tensor result1 = OpPreparation::ApplyTensor(self, output_size);
  at::Tensor result2 = OpPreparation::ApplyTensor(self, output_size);

  int64_t real_correction = 1;
  bool unbiased = true;
  if (correction.has_value()) {
    real_correction = correction.value();
    unbiased = real_correction != 0;
  }
  std_mean_out_npu_nocheck(result1, result2, self, dims, unbiased, keepdim, real_correction);

  return std::tie(result1, result2);
}

at::Tensor NPUNativeFunctions::std(
    const at::Tensor & self,
    at::DimnameList dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  return NPUNativeFunctions::std(self, dimnames_to_positions(self, dim), correction, keepdim);
}

at::Tensor NPUNativeFunctions::std(
    const at::Tensor & self,
    at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  c10::SmallVector<int64_t, SIZE> dims = CalcuOpUtil::GetDimlistForTensor(self);
  if (dim.has_value()) {
    dims = array_to_small_vector(dim.value());
  }

  auto output_size = reduce_ops_npu_output_size(self, dims, keepdim);

  at::Tensor result1 = OpPreparation::ApplyTensor(self, output_size);
  at::Tensor result2 = OpPreparation::ApplyTensor(self, output_size);

  int64_t real_correction = 1;
  bool unbiased = true;
  if (correction.has_value()) {
    real_correction = correction.value();
    unbiased = real_correction != 0;
  }
  std_mean_out_npu_nocheck(result1, result2, self, dims, unbiased, keepdim, real_correction);

  return result1;
}

} // namespace native
} // namespace at_npu
