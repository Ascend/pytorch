// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor &sub_out_npu_nocheck(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar alpha,
    at::Tensor &result) {
  // executing the NPU operator
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    c10::Scalar others = at_npu::native::CalcuOpUtil::ConvertTensorToScalar(other);
    EXEC_NPU_CMD(aclnnSubs, self, others, alpha, result);
  } else {
    EXEC_NPU_CMD(aclnnSub, self, other, alpha, result);
  }
  return result;
}

static at::Tensor self_tensor_to_device(const at::Tensor &tensor, const at::ScalarType resultType) {
  if (at_npu::native::CalcuOpUtil::IsScalarWrappedToTensor(tensor)) {
    at::Scalar scalar = CalcuOpUtil::ConvertTensorToScalar(tensor);
    return CalcuOpUtil::CopyScalarToDevice(scalar, resultType);
  }
  return tensor;
}

at::Tensor &NPUNativeOpApiFunctions::sub_out(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &result) {
  DO_COMPATIBILITY(aclnnSub, NPUNativeFunctions::sub_out(self, other, alpha, result));
  DO_COMPATIBILITY(aclnnSubs, NPUNativeFunctions::sub_out(self, other, alpha, result));
  bool is_self_wrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  at::Tensor output_tensor = is_self_wrapped ? other : self;
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  at::Tensor self_converted = self_tensor_to_device(self, result_type);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(output_tensor),
      result.scalar_type(),
      output_size);

  sub_out_npu_nocheck(self_converted, other, alpha, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::sub(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha) {
  DO_COMPATIBILITY(aclnnSub, NPUNativeFunctions::sub(self, other, alpha));
  DO_COMPATIBILITY(aclnnSubs, NPUNativeFunctions::sub(self, other, alpha));
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  at::Tensor self_converted = self_tensor_to_device(self, result_type);

  auto result = OpPreparation::ApplyTensor(output_size, self.options(), self);
  sub_out_npu_nocheck(self_converted, other, alpha, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::sub(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha) {
  DO_COMPATIBILITY(aclnnSubs, NPUNativeFunctions::sub(self, other, alpha));
  auto output_size = input_same_output_size(self);
  auto result = OpPreparation::ApplyTensor(output_size, self.options(), self);
  EXEC_NPU_CMD(aclnnSubs, self, other, alpha, result);
  return result;
}

} // namespace native
} // namespace at_npu
