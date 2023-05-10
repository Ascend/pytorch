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
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

c10::SmallVector<int64_t, SIZE> kthvalue_npu_output_size(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  at::IntArrayRef dims(dim);
  return reduce_ops_npu_output_size(self, dims, keepdim);
}

void kthvalue_shape_modify(
    at::Tensor& values, 
    at::Tensor& indices, 
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  at::Tensor _self = self.rename(c10::nullopt);
  auto outputSize = kthvalue_npu_output_size(self, dim, keepdim);
  if (values.defined()) {
    TORCH_CHECK(
        values.dtype() == self.dtype(),
        "output values must be of same type as input");
    TORCH_CHECK(
        values.device() == self.device(),
        "output values must be on same values as input");
    values.resize_(outputSize);
  } else {
    values = at::empty(outputSize, _self.options());
  }
  if (indices.defined()) {
    TORCH_CHECK(
        indices.dtype() == at::kLong, 
        "output indices must be of scalar type Long");
    TORCH_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as input");
    indices.resize_(outputSize);
  } else {
    indices = at::empty(outputSize, _self.options().dtype(at::kLong));
  }
  return;
}

void kthvalue_calculate(
    const at::Tensor& self,
    at::Tensor& result,
    at::Tensor x, 
    int64_t k,
    int64_t dim, 
    bool keepdim,
    bool changeType,
    bool isIndices) {
  at::Tensor index = OpPreparation::ApplyTensor(
      {1}, 
      self.options().dtype(at::kInt), 
      self);
  index.fill_(k - 1); 
  at::Tensor y = index_select(x, dim, index);
  if (!keepdim) {
    y.squeeze_(dim);
  }

  if (changeType) {
    y = NPUNativeFunctions::npu_dtype_cast(y, self.scalar_type());
  }
  if (isIndices) {
    y = NPUNativeFunctions::npu_dtype_cast(y, at::kLong);
  }
  result.copy_(y, false);
  at::namedinference::propagate_names_for_reduction(result, self, dim, keepdim);
  return;
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::kthvalue(
    const at::Tensor& self, int64_t k, int64_t dim, bool keepdim) {
  auto outputSize = kthvalue_npu_output_size(self, dim, keepdim);
  at::Tensor values = OpPreparation::ApplyTensor(self, outputSize);
  at::Tensor indices = OpPreparation::ApplyTensorWithFormat(
      outputSize, 
      self.options().dtype(at::kLong), 
      ACL_FORMAT_NCHW);
  kthvalue_out(self, k, dim, keepdim, values, indices);
  return tuple<at::Tensor, at::Tensor>(values, indices);
}

tuple<at::Tensor, at::Tensor>  NPUNativeFunctions::kthvalue(
    const at::Tensor& self, int64_t k, at::Dimname dim, bool keepdim) {
  return kthvalue(self, k, dimname_to_position(self, dim), keepdim);
}

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::kthvalue_out(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim,
    at::Tensor& values,
    at::Tensor& indices) {
  at::SmallVector<int64_t, SIZE> dims = {dim };
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  OpPreparation::CheckOut(
      {self},
      values,
      CalcuOpUtil::GetTensorNpuFormat(self),
      self.scalar_type(),
      outputSize);
  OpPreparation::CheckOut(
      {self},
      indices,
      ACL_FORMAT_ND,
      at::ScalarType::Long,
      outputSize);
  TORCH_CHECK(
      self.scalar_type() == at::kHalf ||
      self.scalar_type() == at::kFloat ||
      self.scalar_type() == at::kInt,
      "the type of input must be float16, float32, or int32");
  dim = CalcuOpUtil::MakeWrapDim(dim, self.dim());

  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");

  at::Tensor _self = self.rename(c10::nullopt);
  kthvalue_shape_modify(values, indices, self, dim, keepdim);
  bool changeType = false;
  if (self.scalar_type() != at::kHalf) {
    changeType = true;
    _self = NPUNativeFunctions::npu_dtype_cast(_self, at::kHalf);
  }
  auto ret = at::topk(_self, k, dim, false, true);
  kthvalue_calculate(self, values, std::get<0>(ret), k, dim, keepdim, changeType, false);
  kthvalue_calculate(self, indices, std::get<1>(ret), k, dim, keepdim, false, true);
  return tuple<at::Tensor&, at::Tensor&>(values, indices);
}

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::kthvalue_out(
    const at::Tensor& self,
    int64_t k,
    at::Dimname dim,
    bool keepdim,
    at::Tensor& values,
    at::Tensor& indices) {
  return kthvalue_out(
      self, k, dimname_to_position(self, dim), keepdim, values, indices);
}
} // namespace native
} // namespace at_npu
