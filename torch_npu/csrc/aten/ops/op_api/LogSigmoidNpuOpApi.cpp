// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

tuple<at::Tensor&, at::Tensor&> NPUNativeOpApiFunctions::log_sigmoid_forward_out(
    const at::Tensor &self,
    at::Tensor &out,
    at::Tensor &buffer) {
  DO_COMPATIBILITY(aclnnLogSigmoid, NPUNativeFunctions::log_sigmoid_forward_out(self, out, buffer));
  OpPreparation::CheckOut({self}, out, self);
  EXEC_NPU_CMD(aclnnLogSigmoidForward, self, out, buffer);
  return std::tie(out, buffer);
}

tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::log_sigmoid_forward(const at::Tensor &self) {
  DO_COMPATIBILITY(aclnnLogSigmoid, NPUNativeFunctions::log_sigmoid_forward(self));
  at::Tensor out = OpPreparation::ApplyTensorWithoutFormat(self);
  at::Tensor buffer = OpPreparation::ApplyTensorWithSizes({0}, self.options());
  EXEC_NPU_CMD(aclnnLogSigmoidForward, self, out, buffer);
  return tuple<at::Tensor, at::Tensor>(out, buffer);
}

at::Tensor &NPUNativeOpApiFunctions::log_sigmoid_out(const at::Tensor &self, at::Tensor &out) {
  DO_COMPATIBILITY(aclnnLogSigmoid, NPUNativeFunctions::log_sigmoid_out(self, out));
  OpPreparation::CheckOut({self}, out, self);
  EXEC_NPU_CMD(aclnnLogSigmoid, self, out);
  return out;
}

at::Tensor NPUNativeOpApiFunctions::log_sigmoid(const at::Tensor &self) {
  DO_COMPATIBILITY(aclnnLogSigmoid, NPUNativeFunctions::log_sigmoid(self));
  at::Tensor out = OpPreparation::ApplyTensorWithoutFormat(self);
  EXEC_NPU_CMD(aclnnLogSigmoid, self, out);
  return out;
}

} // namespace native
} // namespace at_npu
