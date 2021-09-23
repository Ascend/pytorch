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
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<int64_t, SIZE> median_npu_output_size(
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  IntArrayRef dims(dim);
  return reduce_ops_npu_output_size(self, dims, keepdim);
}

Tensor& median_out_npu_nocheck(
    Tensor& result,
    const Tensor& self) {
  // reshape to 1D for global median
  Tensor input = self.has_names() ? 
      self.rename(nullopt).reshape({-1}) : self.reshape({-1});
  int64_t k = input.size(0) / 2;

  auto ret = topk_npu(input, k + 1);
  Tensor topkValues = std::get<0>(ret);
  Tensor value = topkValues[k];
  
  result.fill_(value.item());
  return result;
}

std::tuple<Tensor&, Tensor&> median_out_npu_nocheck(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  int64_t k = self.dim() > 0 ? (self.size(dim) + 1) / 2 : 1;
  
  // drop names, npu_transpose is not yet supported with named tensors
  Tensor _self = self.has_names() ? self.rename(nullopt) : self;
  auto ret = topk_npu(_self, k, dim, false, true);
  Tensor topkValues = std::get<0>(ret);
  Tensor topkIndices = std::get<1>(ret);

  if (topkIndices.dtype() == ScalarType::Long) {
    topkIndices = topkIndices.to(at::kInt);
  }

  Tensor index = at::empty_with_format(
      {1}, _self.options().dtype(kInt), ACL_FORMAT_NCHW);
  index.fill_(k - 1);
  Tensor _values = index_select_npu(topkValues, dim, index);
  Tensor _indices = index_select_npu(topkIndices, dim, index);
  if (!keepdim) {
    _values.squeeze_(dim);
    _indices.squeeze_(dim);
  }
  // add names, copy from kthvalue_out_cpu
  namedinference::propagate_names_for_reduction(_values, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(_indices, self, dim, keepdim);

  copy_npu_(values, _values);
  copy_npu_(indices, _indices);
  return tuple<Tensor&, Tensor&>(values, indices);
}

std::tuple<Tensor&, Tensor&> median_out_npu_nocheck(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    Dimname dim,
    bool keepdim) {
  return median_out_npu_nocheck(values, indices, self, dimname_to_position(self, dim), keepdim);
}

std::tuple<Tensor&, Tensor&> median_out_npu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
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
    ScalarType::Int, 
    outputSize);

  median_out_npu_nocheck(values, indices, self, dim, keepdim);
  return tuple<Tensor&, Tensor&>(values, indices);
}

std::tuple<Tensor&, Tensor&> median_out_npu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    Dimname dim,
    bool keepdim) {
    return median_out_npu(values, indices, self, dimname_to_position(self, dim), keepdim);
}

Tensor median_npu(const Tensor& self) {
  // construct the output tensor of the NPU
  // 0D tensor, outputSize = {}
  Tensor result = at::empty_with_format(
      {}, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  median_out_npu_nocheck(result, self);
  return result;
}

std::tuple<Tensor, Tensor> median_npu(
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  // construct the output tensor of the NPU
  auto outputSize = median_npu_output_size(self, dim, keepdim);
  Tensor values = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  Tensor indices = at::empty_with_format(
      outputSize, self.options().dtype(kInt), ACL_FORMAT_NCHW);
  
  // calculate the output result of the NPU
  median_out_npu_nocheck(values, indices, self, dim, keepdim);
  return tuple<Tensor&, Tensor&>(values, indices);
}

std::tuple<Tensor, Tensor> median_npu(
    const Tensor& self,
    Dimname dim,
    bool keepdim) {
  return median_npu(self, dimname_to_position(self, dim), keepdim);
}

} // namespace native
} // namespace at