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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<int64_t, SIZE> kthvalue_npu_output_size(
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  IntArrayRef dims(dim);
  return reduce_ops_npu_output_size(self, dims, keepdim);
}

void kthvalue_shape_modify(
    Tensor& values, 
    Tensor& indices, 
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  Tensor _self = self.rename(nullopt);

  // Calculate the shape of the output tensor.
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
        indices.dtype() == kLong, 
        "output indices must be of scalar type Long");
    TORCH_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as input");
    indices.resize_(outputSize);
  } else {
    indices = at::empty(outputSize, _self.options().dtype(kLong));
  }

  return;
}

void kthvalue_calculate(
    const Tensor& self,
    Tensor& result,
    Tensor x, 
    int64_t k,
    int64_t dim, 
    bool keepdim,
    bool changeType,
    bool isIndices) {
  Tensor index = at::empty_with_format(
      {1}, 
      self.options().dtype(kInt), 
      CalcuOpUtil::get_tensor_npu_format(self));
  index.fill_(k - 1); 
  Tensor y = index_select_npu(x, dim, index);
  if (!keepdim) {
    y.squeeze_(dim);
  }

  if (changeType) {
    y = y.npu_dtype_cast(self.scalar_type());
  }
  if (isIndices) {
    y = y.npu_dtype_cast(at::kLong);
  }
  
  result = copy_npu_(result, y, false);
  // Add names.
  namedinference::propagate_names_for_reduction(result, self, dim, keepdim);

  return;
}

tuple<Tensor, Tensor> kthvalue_npu(
    const Tensor& self, int64_t k, int64_t dim, bool keepdim) {
  auto outputSize = kthvalue_npu_output_size(self, dim, keepdim);
  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);

  Tensor values = at::empty_with_format(
      outputSize, self.options(), npu_format);
  Tensor indices = at::empty_with_format(
      outputSize, self.options().dtype(kLong), ACL_FORMAT_NCHW);

  kthvalue_out_npu(values, indices, self, k, dim, keepdim);
  return tuple<Tensor, Tensor>(values, indices);
}

tuple<Tensor, Tensor>  kthvalue_npu(
    const Tensor& self, int64_t k, Dimname dim, bool keepdim) {
  
  return at::kthvalue(self, k, dimname_to_position(self, dim), keepdim);
}

tuple<Tensor&, Tensor&> kthvalue_out_npu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim) {
  // Check the type of input
  TORCH_CHECK(
      self.scalar_type() == at::kHalf ||
      self.scalar_type() == at::kFloat ||
      self.scalar_type() == at::kInt,
      "the type of input must be float16, float32, or int32");

  // Check whether k meets the requirements.
  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");

  // Drop names, npu_transpose is not yet supported with named tensors.
  Tensor _self = self.rename(nullopt);

  kthvalue_shape_modify(values, indices, self, dim, keepdim);

  bool changeType = false;
  if (self.scalar_type() != at::kHalf) {
    changeType = true;
    _self = _self.npu_dtype_cast(at::kHalf);
  }

  // Get the kth largest tensor.
  auto ret = at::topk(_self, k, dim, false, true);
  kthvalue_calculate(self, values, std::get<0>(ret), k, dim, keepdim, changeType, false);
  kthvalue_calculate(self, indices, std::get<1>(ret), k, dim, keepdim, false, true);

  return tuple<Tensor&, Tensor&>(values, indices);
}

tuple<Tensor&, Tensor&> kthvalue_out_npu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    Dimname dim,
    bool keepdim) {

  return kthvalue_out_npu(
      values, indices, self, k, dimname_to_position(self, dim), keepdim);
}

}
} 