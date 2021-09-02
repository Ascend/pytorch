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

namespace {
  static inline int64_t calculate_prod_output_format(
    const Tensor& self, 
    IntArrayRef size) {
    int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
    // scalar scene no support nz
    if (size.empty()) {
        npu_format = ACL_FORMAT_ND;
    }
    return npu_format;  
  }
}

Tensor& prod_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    SmallVector<int64_t, N> dimList,
    bool keepdim,
    optional<ScalarType> dtype) {
  ScalarType dstType;
  if (dtype.has_value()) {
    dstType = dtype.value();
  } else if (result.defined()) {
    dstType = result.scalar_type();
  } else {
    dstType = self.scalar_type();
  }

  Tensor self_tmp = self;
  if (dstType != self.scalar_type()) {
      self_tmp = self.to(dstType);
  }

  OpCommand cmd;
    cmd.Name("ReduceProd")
    .Input(self_tmp)
    .Input(dimList)
    .Output(result)
    .Attr("keep_dims", keepdim)
    .Run();

  return result;
}

Tensor& prod_out_npu(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  // fp16 transform：fp32 for precise
  if (self.scalar_type() == ScalarType::Half) {
    Tensor result_tmp  = prod_npu(self, dim, keepdim, dtype);
    OpPreparation::CheckOut(
        {result_tmp}, 
        result, 
        ACL_FORMAT_ND, 
        result_tmp.scalar_type(), 
        result_tmp.sizes());
    result.copy_(result_tmp);
    return result;
  } else {
    auto outputSize = prod_npu_output_size(self, dim, keepdim);
    ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();
    OpPreparation::CheckOut({self}, result, ACL_FORMAT_ND, dstType, outputSize);

    prod_out_npu_nocheck(result, self, {dim}, keepdim, dtype);
    return result;
  }
}

Tensor& prod_out_npu(
    Tensor& result,
    const Tensor& self,
    Dimname dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  return prod_out_npu(
      result, self, dimname_to_position(self, dim), keepdim, dtype);
}

Tensor prod_npu(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  Tensor self_tmp = self;
  //Input transform：fp16 to fp32
  if (self.scalar_type() == ScalarType::Half) {
    self_tmp = self.npu_dtype_cast(ScalarType::Float);
  }

  ScalarType dstType = dtype.has_value() ? dtype.value() : self_tmp.scalar_type();

  // calculate the output size
  auto outputSize = prod_npu_output_size(self_tmp, dim, keepdim);

  int64_t npu_format = calculate_prod_output_format(self_tmp, outputSize);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, self_tmp.options().dtype(dstType), npu_format);

  // calculate the output result of the NPU
  prod_out_npu_nocheck(result, self_tmp, {dim}, keepdim, dtype);

  result = result.npu_dtype_cast(self.scalar_type());
  return result;
}

Tensor prod_npu(
    const Tensor& self,
    Dimname dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  return prod_npu(self, dimname_to_position(self, dim), keepdim, dtype);
}

Tensor prod_npu(const Tensor& self, optional<ScalarType> dtype) {
  ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();

  // calculate the output size
  auto outputSize = prod_npu_output_size(self, false);

  int64_t npu_format = calculate_prod_output_format(self, outputSize);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, self.options().dtype(dstType), npu_format);

  // calculate the output result of the NPU
  prod_out_npu_nocheck(
      result, self, CalcuOpUtil::get_dimlist_for_tensor(self), false, dtype);

  return result;
}
} // namespace native
} // namespace at
