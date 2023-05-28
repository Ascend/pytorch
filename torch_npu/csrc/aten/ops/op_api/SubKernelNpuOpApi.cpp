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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>

namespace at_npu {
namespace native {

static at::Tensor wrapper_tensor_cast(const at::Tensor &tensor, const at::ScalarType resultType) {
  if (tensor.scalar_type() != resultType && tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    auto tensor_npu = tensor;
    if(!at_npu::key::isDeviceTensor(tensor)) {
      tensor_npu = CalcuOpUtil::CopyTensorHostToDevice(tensor);
    }
    return NPUNativeFunctions::npu_dtype_cast(tensor_npu, resultType);
  }
  return tensor;
}

at::Tensor &NPUNativeOpApiFunctions::sub_out(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &result) {
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  at::Tensor outputTensor = isSelfWrapped ? other : self;
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::ScalarType resultType = at::native::result_type(self, other);
  at::Tensor selfCasted = wrapper_tensor_cast(self, resultType);
  at::Tensor otherCasted = wrapper_tensor_cast(other, resultType);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(outputTensor),
      result.scalar_type(),
      outputSize);

  EXEC_NPU_CMD(aclnnSub, selfCasted, otherCasted, alpha, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::sub(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha) {
  auto outSize = broadcast_ops_npu_output_size(self, other);
  at::ScalarType resultType = at::native::result_type(self, other);
  at::Tensor selfCasted = wrapper_tensor_cast(self, resultType);
  at::Tensor otherCasted = wrapper_tensor_cast(other, resultType);

  auto result = OpPreparation::ApplyTensorWithFormat(outSize, self.options(), CalcuOpUtil::GetTensorNpuFormat(self));
  EXEC_NPU_CMD(aclnnSub, selfCasted, otherCasted, alpha, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::sub(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha) {
  auto outSize = input_same_output_size(self);
  auto result = OpPreparation::ApplyTensorWithFormat(outSize, self.options(), CalcuOpUtil::GetTensorNpuFormat(self));
  auto otherTensor = CalcuOpUtil::CopyScalarToDevice(other, self.scalar_type());
  EXEC_NPU_CMD(aclnnSub, self, otherTensor, alpha, result);
  return result;
}

at::Tensor &NPUNativeOpApiFunctions::sub_(at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha) {
  at::ScalarType resultType = at::native::result_type(self, other);
  at::Tensor selfCasted = wrapper_tensor_cast(self, resultType);
  at::Tensor otherCasted = wrapper_tensor_cast(other, resultType);

  EXEC_NPU_CMD(aclnnInplaceSub, selfCasted, otherCasted, alpha);
  return self;
}

at::Tensor &NPUNativeOpApiFunctions::sub_(at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha) {
  auto otherTensor = CalcuOpUtil::CopyScalarToDevice(other, self.scalar_type());
  EXEC_NPU_CMD(aclnnInplaceSub, self, otherTensor, alpha);
  return self;
}

} // namespace native
} // namespace at_npu
