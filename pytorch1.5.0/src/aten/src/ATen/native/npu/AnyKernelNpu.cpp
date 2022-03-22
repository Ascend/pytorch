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

inline Tensor& any_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    SmallVector<int64_t, N> dimList,
    bool keepdim) {

  OpCommand cmd;
  cmd.Name("ReduceAny")
    .Input(self)
    .Input(dimList)
    .Output(result)
    .Attr("keep_dims", keepdim)
    .Run();

  return result;
}

Tensor& any_out_npu(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {  
  SmallVector<int64_t, N> dimList;
  if (dim == LLONG_MIN) {
    dimList = CalcuOpUtil::get_dimlist_for_tensor(self);
  } else {
    dimList = {dim};
  }

  // check result for return
  auto outputSize = reduce_ops_npu_output_size(self, dimList, keepdim);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      self.scalar_type(),
      outputSize);

  // calculate the output result of the NPU
  any_out_npu_nocheck(result, self, dimList, keepdim);

  return result;
}

Tensor any_npu(const Tensor& self, int64_t dim, bool keepdim) {
  // calculate the output size
  IntArrayRef dims(dim);
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU  
  if (dim == LLONG_MIN) {
    any_out_npu_nocheck(
        result, self, CalcuOpUtil::get_dimlist_for_tensor(self), keepdim);
  } else {
    any_out_npu_nocheck(result, self, {dim}, keepdim);
  }

  return result;
}

Tensor any_npu(const Tensor& self) { 
  // when self's dim = 0, convert [1] tensor and reduce it
  if (self.dim() == 0) {
      Tensor self_tmp = self;
      self_tmp = at::empty_with_format(
          {1}, 
          self.options().dtype(ScalarType::Float), 
          CalcuOpUtil::get_tensor_npu_format(self))
          .fill_(self.item())
          .to(ScalarType::Bool);
      return any_npu(self_tmp, 0, false);
  }

  // calculate the output size 
  IntArrayRef dims;
  auto outputSize = reduce_ops_npu_output_size(self, dims, false);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  any_out_npu_nocheck(
      result, self, CalcuOpUtil::get_dimlist_for_tensor(self), false);

  return result;
}
} // namespace native
} // namespace at