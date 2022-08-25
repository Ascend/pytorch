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

Tensor& sum_out_npu_no_dtype(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim) {

  SmallVector<int64_t, N> dimList;
  if (dim.empty()) {
    dimList = CalcuOpUtil::get_dimlist_for_tensor(self);
  } else {
    dimList = SmallVector<int64_t, N>(dim);
  }

  OpCommand cmd;
  cmd.Name("ReduceSum")
      .Input(self)
      .Input(dimList, at::kLong)
      .Output(result)
      .Attr("keep_dims", keepdim)
      .Run();
  return result;
}

Tensor& sum_out_npu_int_dtype(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim) {
  Tensor selfs = self.npu_dtype_cast(ScalarType::Float);
  sum_out_npu_no_dtype(result, selfs, dim, keepdim);
  result = result.npu_dtype_cast(ScalarType::Int);
  return result;
}

Tensor& sum_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  ScalarType dstType;
  if (dtype.has_value()) {
    if (dtype.value() == ScalarType::Int) {
      Tensor selfs = self.npu_dtype_cast(ScalarType::Int);
      return sum_out_npu_int_dtype(result, selfs, dim, keepdim);
    } else {
      dstType = dtype.value();
    }
  } else if (isIntegralType(self.scalar_type(), true)) {
    return sum_out_npu_int_dtype(result, self, dim, keepdim);
  } else if (result.defined()) {
    if (isIntegralType(result.scalar_type(), true)) {
      return sum_out_npu_int_dtype(result, self, dim, keepdim);
    } else {
      dstType = result.scalar_type();
    }
  } else {
    dstType = self.scalar_type();
  }
  // dtype same
  if (dstType == self.scalar_type()) {
    sum_out_npu_no_dtype(result, self, dim, keepdim);
    return result;
  }

  sum_out_npu_no_dtype(result, self.toType(dstType), dim, keepdim);
  return result;
}

Tensor& sum_out_npu(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  auto outputSize = sum_npu_output_size(self, dim, keepdim);  
  auto dstType = self.scalar_type();
  if (dtype.has_value()) {
      dstType = dtype.value();
  }

  OpPreparation::CheckOut(
      {self}, 
      result, 
      ACL_FORMAT_ND,
      dstType,
      outputSize);

  OpPipeWithDefinedOut pipe;
  pipe.CheckMemory({self}, {result});

  sum_out_npu_nocheck(result, self, dim, keepdim, dtype);
  return result;
}

Tensor& sum_out_npu(
    Tensor& result,
    const Tensor& self,
    DimnameList dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  return sum_out_npu(
      result, self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor sum_npu(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  ScalarType dstType; 
  if (dtype.has_value()) {
    if(dtype.value() == ScalarType::Int) {
      dstType = ScalarType::Float;
    } else {
      dstType = dtype.value();
    }
  } else if (isIntegralType(self.scalar_type(), true)) {
    dstType = ScalarType::Float;
  } else {
    dstType = self.scalar_type();
  }

  // calculate the output size
  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
  auto selfSize = self.sizes();
  
  for (int64_t i = 0; i < selfSize.size(); i++) {
    if (selfSize[i] == 0) {
      return at::zeros(outputSize, self.options());
    }
  }

  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
  // scalar scene no support nz
  if (outputSize.empty() || outputSize.size() < 4) {
    npu_format = ACL_FORMAT_ND;
  }

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options().dtype(dstType), npu_format);

  // calculate the output result of the NPU
  sum_out_npu_nocheck(result, self, dim, keepdim, dtype);
  return result;
}

Tensor sum_npu(
    const Tensor& self,
    DimnameList dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  return sum_npu(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor sum_npu(const Tensor& self, optional<ScalarType> dtype) {
  return sum_npu(self, SmallVector<int64_t, N>{}, false, dtype);
}

} // namespace native
} // namespace at
