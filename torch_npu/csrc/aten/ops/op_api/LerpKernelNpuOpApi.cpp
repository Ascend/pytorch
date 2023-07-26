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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {

c10::SmallVector<int64_t, SIZE> aclnn_lerp_broadcast_size(
    const at::Tensor& self, 
    const at::Tensor& end, 
    const at::Tensor& weight) {
  auto expanded_size = broadcast_ops_npu_output_size(self, end);
  auto output_size = broadcast_ops_npu_output_size(expanded_size, weight.sizes());
  return output_size;
}

at::Tensor& NPUNativeOpApiFunctions::lerp_out(
    const at::Tensor& self, 
    const at::Tensor& end, 
    const at::Tensor& weight, 
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLerp, NPUNativeFunctions::lerp_out(self, end, weight, result));
  auto output_size = aclnn_lerp_broadcast_size(self, end, weight);
  OpPreparation::CheckOut(
      {self, end, weight},
      result,
      result.scalar_type(),
      output_size);
  EXEC_NPU_CMD(aclnnLerp, self, end, weight, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::lerp_out(
    const at::Tensor& self, 
    const at::Tensor& end, 
    const at::Scalar& weight,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLerps, NPUNativeFunctions::lerp_out(self, end, weight, result));
  auto output_size = broadcast_ops_npu_output_size(self, end);
  OpPreparation::CheckOut(
      {self, end},
      result,
      result.scalar_type(),
      output_size);
  EXEC_NPU_CMD(aclnnLerps, self, end, weight, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::lerp(const at::Tensor& self, const at::Tensor& end, const at::Tensor& weight) {
  DO_COMPATIBILITY(aclnnLerp, NPUNativeFunctions::lerp(self, end, weight));
  auto output_size = aclnn_lerp_broadcast_size(self, end, weight);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  EXEC_NPU_CMD(aclnnLerp, self, end, weight, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::lerp(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight) {
  DO_COMPATIBILITY(aclnnLerps, NPUNativeFunctions::lerp(self, end, weight));
  auto output_size = broadcast_ops_npu_output_size(self, end);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  EXEC_NPU_CMD(aclnnLerps, self, end, weight, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::lerp_(at::Tensor& self, const at::Tensor& end, const at::Tensor& weight) {
  DO_COMPATIBILITY(aclnnInplaceLerp, NPUNativeFunctions::lerp_(self, end, weight));
  EXEC_NPU_CMD(aclnnInplaceLerp, self, end, weight);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::lerp_(at::Tensor& self, const at::Tensor& end, const at::Scalar& weight) {
  DO_COMPATIBILITY(aclnnInplaceLerps, NPUNativeFunctions::lerp_(self, end, weight));
  EXEC_NPU_CMD(aclnnInplaceLerps, self, end, weight);
  return self;
}
} // namespace native
} // namespace at_npu
