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

#include <ATen/Tensor.h>
#include <c10/util/SmallVector.h>

#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace at_npu {
namespace native {

inline void alpha_check_npu(const at::ScalarType dtype, at::Scalar alpha) {
  TORCH_CHECK(!alpha.isBoolean() || dtype == at::ScalarType::Bool, "Boolean alpha only supported for Boolean results.");
  TORCH_CHECK(isFloatingType(dtype) || alpha.isIntegral(true),
              "For integral input tensors, argument alpha must not be a floating point number.");
}

static at::Tensor& add_out_npu_nocheck(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha,
                                       at::Tensor& result) {
  // executing the NPU operator
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    c10::Scalar others = at_npu::native::CalcuOpUtil::ConvertTensorToScalar(other);
    EXEC_NPU_CMD(aclnnAdds, self, others, alpha, result);
  } else {
    EXEC_NPU_CMD(aclnnAdd, self, other, alpha, result);
  }
  return result;
}

static at::Tensor self_tensor_to_device(const at::Tensor& tensor, const at::ScalarType result_type) {
  if (at_npu::native::CalcuOpUtil::IsScalarWrappedToTensor(tensor)) {
    at::Scalar scalar = CalcuOpUtil::ConvertTensorToScalar(tensor);
    return CalcuOpUtil::CopyScalarToDevice(scalar, result_type);
  }
  return tensor;
}

static at::Tensor add_dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  return isSelfWrapped ? other : self;
}

at::Tensor NPUNativeOpApiFunctions::add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  DO_COMPATIBILITY(aclnnAdd, NPUNativeFunctions::add(self, other, alpha));
  DO_COMPATIBILITY(aclnnAdds, NPUNativeFunctions::add(self, other, alpha));
  alpha_check_npu(self.scalar_type(), alpha);
  // calculate the output size
  at::Tensor output_tensor = add_dest_output(self, other);
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  at::Tensor self_cp = self_tensor_to_device(self, result_type);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(output_size, output_tensor.options().dtype(result_type),
                                                           CalcuOpUtil::GetTensorNpuFormat(output_tensor));

  // calculate the output result of the NPU
  add_out_npu_nocheck(self_cp, other, alpha, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::add(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
  DO_COMPATIBILITY(aclnnAdds, NPUNativeFunctions::add(self, other, alpha));
  alpha_check_npu(self.scalar_type(), alpha);
  // calculate the output size
  auto output_size = input_same_output_size(self);
  at::ScalarType result_type = at::native::result_type(self, other);
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(output_size, self.options().dtype(result_type),
                                                           CalcuOpUtil::GetTensorNpuFormat(self));
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnAdds, self, other, alpha, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::add_out(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha,
                                             at::Tensor& result) {
  DO_COMPATIBILITY(aclnnAdd, NPUNativeFunctions::add_out(self, other, alpha, result));
  DO_COMPATIBILITY(aclnnAdds, NPUNativeFunctions::add_out(self, other, alpha, result));
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  at::Tensor output_tensor = isSelfWrapped ? other : self;
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  at::Tensor self_cp = self_tensor_to_device(self, result_type);

  OpPreparation::CheckOut({self}, result, CalcuOpUtil::GetTensorNpuFormat(output_tensor), result.scalar_type(),
                          output_size);

  CalcuOpUtil::CheckMemoryOverLaps({self, other}, {result});
  add_out_npu_nocheck(self_cp, other, alpha, result);
  return result;
}

}  // namespace native
}  // namespace at_npu
