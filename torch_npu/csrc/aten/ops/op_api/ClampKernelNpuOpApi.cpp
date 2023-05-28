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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>

namespace at_npu {
namespace native {
at::Tensor& NPUNativeOpApiFunctions::clamp_out(
    const at::Tensor& self,
    const c10::optional<at::Scalar>& min,
    const c10::optional<at::Scalar>& max,
    at::Tensor& result) {
  TORCH_CHECK(min.has_value() || max.has_value(), "torch.clamp: At least one of 'min' or 'max' must not be None");
  EXEC_NPU_CMD(aclnnClamp, self, min, max, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::clamp(
    const at::Tensor& self,
    const c10::optional<at::Scalar>& min,
    const c10::optional<at::Scalar>& max) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  return NPUNativeOpApiFunctions::clamp_out(self, min, max, result);
}

at::Tensor& NPUNativeOpApiFunctions::clamp_out(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& min,
    const c10::optional<at::Tensor>& max,
    at::Tensor& result) {
  TORCH_CHECK(min.has_value() || max.has_value(), "torch.clamp: At least one of 'min' or 'max' must not be None");
  EXEC_NPU_CMD(aclnnClampTensor, self, min, max, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::clamp(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& min,
    const c10::optional<at::Tensor>& max) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  return NPUNativeOpApiFunctions::clamp_out(self, min, max, result);
}

at::Tensor& NPUNativeOpApiFunctions::clamp_min_out(
    const at::Tensor& self, 
    const at::Scalar& min,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  EXEC_NPU_CMD(aclnnClampMin, self, min, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::clamp_min(const at::Tensor& self, const at::Scalar& min) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  return NPUNativeOpApiFunctions::clamp_min_out(self, min, result);
}

} // namespace native
} // namespace at_npu
