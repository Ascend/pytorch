// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {
at::Tensor& NPUNativeOpApiFunctions::clamp_out(const at::Tensor& self, const c10::optional<at::Scalar>& min,
                                               const c10::optional<at::Scalar>& max, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnClamp, NPUNativeFunctions::clamp_out(self, min, max, result));
  OpPreparation::CheckOut({self}, result, self.scalar_type(), self.sizes());
  EXEC_NPU_CMD(aclnnClamp, self, min, max, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::clamp(const at::Tensor& self, const c10::optional<at::Scalar>& min,
                                          const c10::optional<at::Scalar>& max) {
  DO_COMPATIBILITY(aclnnClamp, NPUNativeFunctions::clamp(self, min, max));
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self);
  return NPUNativeOpApiFunctions::clamp_out(self, min, max, result);
}

at::Tensor& NPUNativeOpApiFunctions::clamp_(at::Tensor& self, const c10::optional<at::Scalar>& min,
                                            const c10::optional<at::Scalar>& max) {
  DO_COMPATIBILITY(aclnnClamp, NPUNativeFunctions::clamp_(self, min, max));
  return NPUNativeOpApiFunctions::clamp_out(self, min, max, self);
}

at::Tensor& NPUNativeOpApiFunctions::clamp_out(const at::Tensor& self, const c10::optional<at::Tensor>& min,
                                               const c10::optional<at::Tensor>& max, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnClampTensor, NPUNativeFunctions::clamp_out(self, min, max, result));
  auto out_size = clamp_npu_output_size(self, min, max);
  OpPreparation::CheckOut({self}, result, result.scalar_type(), out_size);
  EXEC_NPU_CMD(aclnnClampTensor, self, min, max, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::clamp_(at::Tensor& self, const c10::optional<at::Tensor>& min,
                                            const c10::optional<at::Tensor>& max) {
  DO_COMPATIBILITY(aclnnClampTensor, NPUNativeFunctions::clamp_(self, min, max));
  return NPUNativeOpApiFunctions::clamp_out(self, min, max, self);
}

at::Tensor NPUNativeOpApiFunctions::clamp(const at::Tensor& self, const c10::optional<at::Tensor>& min,
                                          const c10::optional<at::Tensor>& max) {
  DO_COMPATIBILITY(aclnnClampTensor, NPUNativeFunctions::clamp(self, min, max));
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, clamp_npu_output_size(self, min, max));
  EXEC_NPU_CMD(aclnnClampTensor, self, min, max, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::clamp_min_out(const at::Tensor& self, const at::Scalar& min, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnClampMin, NPUNativeFunctions::clamp_min_out(self, min, result));
  OpPreparation::CheckOut({self}, result, self.scalar_type(), self.sizes());
  EXEC_NPU_CMD(aclnnClampMin, self, min, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::clamp_min(const at::Tensor& self, const at::Scalar& min) {
  DO_COMPATIBILITY(aclnnClampMin, NPUNativeFunctions::clamp_min(self, min));
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self);
  return NPUNativeOpApiFunctions::clamp_min_out(self, min, result);
}

at::Tensor& NPUNativeOpApiFunctions::clamp_min_(at::Tensor& self, const at::Scalar& min) {
  DO_COMPATIBILITY(aclnnClampMin, NPUNativeFunctions::clamp_min_(self, min));
  return NPUNativeOpApiFunctions::clamp_min_out(self, min, self);
}

at::Tensor& NPUNativeOpApiFunctions::clamp_max_out(const at::Tensor& self, const at::Tensor& max, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnClampMaxTensor, NPUNativeFunctions::clamp_max_out(self, max, result));
  auto output_size = broadcast_ops_npu_output_size(self, max);
  OpPreparation::CheckOut({self, max}, result, result.scalar_type(), output_size);
  EXEC_NPU_CMD(aclnnClampMaxTensor, self, max, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::clamp_max(const at::Tensor& self, const at::Tensor& max) {
  DO_COMPATIBILITY(aclnnClampMaxTensor, NPUNativeFunctions::clamp_max(self, max));
  auto output_size = broadcast_ops_npu_output_size(self, max);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  EXEC_NPU_CMD(aclnnClampMaxTensor, self, max, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::clamp_min_out(const at::Tensor& self, const at::Tensor& min, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnClampMinTensor, NPUNativeFunctions::clamp_min_out(self, min, result));
  OpPreparation::CheckOut({self}, result, self.scalar_type(), self.sizes());
  EXEC_NPU_CMD(aclnnClampMinTensor, self, min, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::clamp_min(const at::Tensor& self, const at::Tensor& min) {
  DO_COMPATIBILITY(aclnnClampMinTensor, NPUNativeFunctions::clamp_min(self, min));
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self);
  EXEC_NPU_CMD(aclnnClampMinTensor, self, min, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::clamp_min_(at::Tensor& self, const at::Tensor& min) {
  DO_COMPATIBILITY(aclnnInplaceClampMinTensor, NPUNativeFunctions::clamp_min_(self, min));
  EXEC_NPU_CMD(aclnnInplaceClampMinTensor, self, min);
  return self;
}

at::Tensor NPUNativeOpApiFunctions::clamp_max(const at::Tensor& self, const at::Scalar& max) {
  DO_COMPATIBILITY(aclnnClampMax, NPUNativeFunctions::clamp_max(self, max));

  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self);
  EXEC_NPU_CMD(aclnnClampMax, self, max, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::clamp_max_(at::Tensor& self, const at::Scalar& max) {
  DO_COMPATIBILITY(aclnnInplaceClampMax, NPUNativeFunctions::clamp_max_(self, max));

  EXEC_NPU_CMD(aclnnInplaceClampMax, self,max);
  return self;
}
}  // namespace native
}  // namespace at_npu
