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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
at::Tensor& NPUNativeOpApiFunctions::mean_out(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                                              c10::optional<c10::ScalarType> dtype, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnMean, NPUNativeFunctions::mean_out(self, dim, keepdim, dtype, result));
  c10::ScalarType dstType;
  if (dtype.has_value()) {
    dstType = dtype.value();
  } else if (result.defined()) {
    dstType = result.scalar_type();
  } else {
    dstType = self.scalar_type();
  }
  // 推导reduecshape
  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
  int64_t npu_format = CalcuOpUtil::GetTensorNpuFormat(result);
  OpPreparation::CheckOut({self}, result, npu_format, result.scalar_type(), outputSize);

  EXEC_NPU_CMD(aclnnMean, self, dim, keepdim, dstType, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::mean(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                                         c10::optional<c10::ScalarType> dtype) {
  DO_COMPATIBILITY(aclnnMean, NPUNativeFunctions::mean(self, dim, keepdim, dtype));
  c10::ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();

  // calculate the output size
  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);

  int64_t npu_format = CalcuOpUtil::GetTensorNpuFormat(self);
  // scalar scene no support nz
  if (outputSize.empty()) {
    npu_format = ACL_FORMAT_NCHW;
  }

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(outputSize, self.options().dtype(dstType), npu_format);

  // calculate the output result of the NPU
  NPUNativeOpApiFunctions::mean_out(self, dim, keepdim, dtype, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::mean(const at::Tensor &self, at::DimnameList dim, bool keepdim,
                                    c10::optional<c10::ScalarType> dtype)
{
  DO_COMPATIBILITY(aclnnMean, NPUNativeFunctions::mean(self, dim, keepdim, dtype));
  return NPUNativeOpApiFunctions::mean(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

at::Tensor &NPUNativeOpApiFunctions::mean_out(const at::Tensor &self, at::DimnameList dim, bool keepdim,
                                         c10::optional<c10::ScalarType> dtype, at::Tensor &result)
{
  DO_COMPATIBILITY(aclnnMean, NPUNativeFunctions::mean_out(self, dim, keepdim, dtype, result));
  return NPUNativeOpApiFunctions::mean_out(self, dimnames_to_positions(self, dim), keepdim, dtype, result);
}

at::Tensor NPUNativeOpApiFunctions::mean(const at::Tensor& self, c10::optional<c10::ScalarType> dtype) {
  DO_COMPATIBILITY(aclnnMean, NPUNativeFunctions::mean(self, dtype));
  return NPUNativeOpApiFunctions::mean(self, c10::SmallVector<int64_t, N>{}, false, dtype);
}

}  // namespace native
}  // namespace at_npu
