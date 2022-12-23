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

c10::SmallVector<int64_t, SIZE> median_npu_output_size(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  at::IntArrayRef dims(dim);
  return reduce_ops_npu_output_size(self, dims, keepdim);
}

at::Tensor& median_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self) {
  at::Tensor input = self.has_names() ?
      self.rename(c10::nullopt).reshape({-1}) : self.reshape({-1});
  int64_t k = input.size(0) / 2;

  auto ret = topk(input, k + 1);
  at::Tensor topkValues = std::get<0>(ret);
  at::Tensor value = topkValues[k];

  result.fill_(value);
  return result;
}

std::tuple<at::Tensor&, at::Tensor&> median_out_value_nocheck(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  int64_t k = self.dim() > 0 ? (self.size(dim) + 1) / 2 : 1;

  at::Tensor _self = self.has_names() ? self.rename(c10::nullopt) : self;
  auto ret = topk(_self, k, dim, false, true);
  at::Tensor topkValues = std::get<0>(ret);
  at::Tensor topkIndices = std::get<1>(ret);

  //NCHW -> reflush base format
  at::Tensor index = OpPreparation::ApplyTensorWithFormat(
      {1}, _self.options().dtype(at::kLong), ACL_FORMAT_NCHW);
  index.fill_(k - 1);
  at::Tensor _values = index_select(topkValues, dim, index);
  at::Tensor _indices = index_select(topkIndices, dim, index);
  if (!keepdim) {
    _values.squeeze_(dim);
    _indices.squeeze_(dim);
  }
  at::namedinference::propagate_names_for_reduction(_values, self, dim, keepdim);
  at::namedinference::propagate_names_for_reduction(_indices, self, dim, keepdim);
  values.copy_(_values);
  indices.copy_(_indices);
  return tuple<at::Tensor&, at::Tensor&>(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> median_out_npu_nocheck(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    at::Dimname dim,
    bool keepdim) {
  return median_out_value_nocheck(values, indices, self, dimname_to_position(self, dim), keepdim);
}

std::tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::median_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& values,
    at::Tensor& indices) {
  auto outputSize = median_npu_output_size(self, dim, keepdim);
  OpPreparation::CheckOut(
      {self},
      values,
      ACL_FORMAT_ND,
      self.scalar_type(),
      outputSize);

  OpPreparation::CheckOut(
      {self},
      indices,
      ACL_FORMAT_ND,
      at::ScalarType::Long,
      outputSize);

  median_out_value_nocheck(values, indices, self, dim, keepdim);
  return tuple<at::Tensor&, at::Tensor&>(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::median_out(
    const at::Tensor& self,
    at::Dimname dim,
    bool keepdim,
    at::Tensor& values,
    at::Tensor& indices) {
    return at::median_out(values, indices, self, dimname_to_position(self, dim), keepdim);
}

at::Tensor NPUNativeFunctions::median(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      {}, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  median_out_nocheck(result, self);
  return result;
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::median(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto outputSize = median_npu_output_size(self, dim, keepdim);
  at::Tensor values = OpPreparation::ApplyTensorWithFormat(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  //NCHW -> reflush base format
  at::Tensor indices = OpPreparation::ApplyTensorWithFormat(
      outputSize, self.options().dtype(at::kLong), ACL_FORMAT_NCHW);

  median_out_value_nocheck(values, indices, self, dim, keepdim);
  return tuple<at::Tensor&, at::Tensor&>(values, indices);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::median(
    const at::Tensor& self,
    at::Dimname dim,
    bool keepdim) {
  return median(self, dimname_to_position(self, dim), keepdim);
}

} // namespace native
} // namespace at_npu
