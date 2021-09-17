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

Tensor& mean_out_npu_no_dtype_nocheck(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim) {

  if (self.numel()==0 && dim.size()==0) {
    // In this scenario, needs to return nan. And the nan of the NPU can only be fp32.
    result = result.to(at::kFloat).fill_(0);
    result = result / 0;
    return result;
  }

  SmallVector<int64_t, N> dimVec;
  if (dim.empty()) {
    dimVec = CalcuOpUtil::get_dimlist_for_tensor(self);
  } else {
    dimVec = array_to_small_vector(dim);
  }

  OpCommand cmd;
  cmd.Name("ReduceMean")
    .Input(self)
    .Input(dimVec, at::kLong)
    .Output(result)
    .Attr("keep_dims",keepdim)
    .Run();
  return result;
}

Tensor& mean_out_npu_no_dtype(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim) {
  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(result);
  if (outputSize.empty()) {
    npu_format = ACL_FORMAT_NCHW;
  }
  OpPreparation::CheckOut(
      {self},
      result,
      npu_format,
      self.scalar_type(),
      outputSize);

  mean_out_npu_no_dtype_nocheck(result, self, dim, keepdim);
  return result;
}

Tensor& mean_out_npu(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& result) {
  ScalarType dstType;
  if (dtype.has_value()) {
    dstType = dtype.value();
  } else if (result.defined()) {
    dstType = result.scalar_type();
  } else {
    dstType = self.scalar_type();
  }

  // dtype same
  if (dstType == self.scalar_type()) {
    mean_out_npu_no_dtype(result, self, dim, keepdim);
    return result;
  }

  mean_out_npu_no_dtype(result, self.toType(dstType), dim, keepdim);
  return result;
}

Tensor& mean_dimlist_out_npu(
    const Tensor& self,
    DimnameList dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& result) {
  return mean_out_npu(self, dimnames_to_positions(self, dim), keepdim, dtype, result);
}

Tensor mean_dim_npu(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();

  // calculate the output size
  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);

  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
  // scalar scene no support nz
  if (outputSize.empty()) {
    npu_format = ACL_FORMAT_NCHW;
  }

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options().dtype(dstType), npu_format);

  // calculate the output result of the NPU
  mean_out_npu(self, dim, keepdim, dtype, result);
  return result;
}

Tensor mean_dimlist_npu(
    const Tensor& self,
    DimnameList dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  return mean_dim_npu(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor mean_npu(const Tensor& self, optional<ScalarType> dtype) {
  return mean_dim_npu(self, SmallVector<int64_t, N>{}, false, dtype);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("mean", TORCH_FN(mean_npu));
  m.impl("mean.dim", TORCH_FN(mean_dim_npu));
  m.impl("mean.out", TORCH_FN(mean_out_npu));
  m.impl("mean.names_dim", TORCH_FN(mean_dimlist_npu));
  m.impl("mean.names_out", TORCH_FN(mean_dimlist_out_npu));
}
} // namespace native
} // namespace at