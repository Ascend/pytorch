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

namespace {
static inline int64_t calculate_prod_output_format(
    const at::Tensor& self,
    at::IntArrayRef size) {
  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
  // scalar scene no support nz
  if (size.empty()) {
    npu_format = ACL_FORMAT_ND;
  }
  return npu_format;
}
}

at::Tensor& prod_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    c10::SmallVector<int64_t, N> dimList,
    bool keepdim,
    c10::optional<at::ScalarType> dtype) {
  OpCommand cmd;
    cmd.Name("ReduceProd")
    .Input(self)
    .Input(dimList)
    .Output(result)
    .Attr("keep_dims", keepdim)
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::prod_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& result) {
  at::Tensor self_tmp = self;
  // fp16 transform：fp32 for precise
  if (self.scalar_type() == at::ScalarType::Half) {
    self_tmp = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }

  auto outputSize = prod_npu_output_size(self, dim, keepdim);
  at::ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();

  OpPreparation::CheckOut(
      {self_tmp},
      result,
      ACL_FORMAT_ND,
      dstType,
      outputSize);

  at::Tensor result_tmp = result;
  if (result_tmp.scalar_type() == at::ScalarType::Half) {
    result_tmp = NPUNativeFunctions::npu_dtype_cast(result_tmp, at::ScalarType::Float);
  }

  prod_out_npu_nocheck(result_tmp, self_tmp, {dim}, keepdim, dtype);

  if (result_tmp.scalar_type() != dstType) {
    result_tmp = NPUNativeFunctions::npu_dtype_cast(result_tmp, dstType);
  }
  result.copy_(result_tmp);

  return result;
}

at::Tensor& NPUNativeFunctions::prod_out(
    const at::Tensor& self,
    at::Dimname dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& result) {
  return prod_out(
      self, dimname_to_position(self, dim), keepdim, dtype, result);
}

at::Tensor NPUNativeFunctions::prod(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype) {
  at::Tensor self_tmp = self;
  // Input transform：fp16 to fp32
  if (self.scalar_type() == at::ScalarType::Half) {
    self_tmp = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }

  at::ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();

  // calculate the output size
  auto outputSize = prod_npu_output_size(self_tmp, dim, keepdim);

  int64_t npu_format = calculate_prod_output_format(self_tmp, outputSize);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, self_tmp.options(), npu_format);

  // calculate the output result of the NPU
  prod_out_npu_nocheck(result, self_tmp, {dim}, keepdim, dtype);

  if (result.scalar_type() != dstType) {
    result = NPUNativeFunctions::npu_dtype_cast(result, dstType);
  }

  return result;
}

at::Tensor NPUNativeFunctions::prod(
    const at::Tensor& self,
    at::Dimname dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype) {
  return prod(self, dimname_to_position(self, dim), keepdim, dtype);
}

at::Tensor NPUNativeFunctions::prod(const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  at::Tensor self_tmp = self;
  // Input transform：fp16 to fp32
  if (self.scalar_type() == at::ScalarType::Half) {
    self_tmp = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }

  at::ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();

  // calculate the output size
  auto outputSize = prod_npu_output_size(self, false);

  int64_t npu_format = calculate_prod_output_format(self, outputSize);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, self_tmp.options(), npu_format);

  // calculate the output result of the NPU
  prod_out_npu_nocheck(
      result, self_tmp, CalcuOpUtil::get_dimlist_for_tensor(self), false, dtype);

  if (result.scalar_type() != dstType) {
    result = npu_dtype_cast(result, dstType);
  }

  return result;
}
} // namespace native
} // namespace at_npu