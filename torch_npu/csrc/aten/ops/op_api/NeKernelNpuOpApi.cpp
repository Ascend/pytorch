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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::ne_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnNeTensor, NPUNativeFunctions::ne_out(self, other, result));
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  at::Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut({self, other}, result, CalcuOpUtil::GetTensorNpuFormat(formatCastOfSelf),
                          result.scalar_type(), at::IntArrayRef(outputSize));
  EXEC_NPU_CMD(aclnnNeTensor, formatCastOfSelf, formatCastOfOther, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::ne_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnNeScalar, NPUNativeFunctions::ne_out(self, other, result));
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  OpPreparation::CheckOut({self}, result, CalcuOpUtil::GetTensorNpuFormat(formatCastOfSelf), result.scalar_type(),
                          formatCastOfSelf.sizes());
  EXEC_NPU_CMD(aclnnNeScalar, formatCastOfSelf, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::ne(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnNeTensor, NPUNativeFunctions::ne(self, other));
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  at::Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);

  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);
  at::Tensor result =
      OpPreparation::ApplyTensor(outputSize, formatCastOfSelf.options().dtype(at::kBool), formatCastOfSelf);

  EXEC_NPU_CMD(aclnnNeTensor, formatCastOfSelf, formatCastOfOther, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::ne(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnNeScalar, NPUNativeFunctions::ne(self, other));
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);

  at::Tensor result = OpPreparation::ApplyTensor(formatCastOfSelf, formatCastOfSelf.options().dtype(at::kBool));

  EXEC_NPU_CMD(aclnnNeScalar, formatCastOfSelf, other, result);
  return result;
}

}  // namespace native
}  // namespace at_npu
