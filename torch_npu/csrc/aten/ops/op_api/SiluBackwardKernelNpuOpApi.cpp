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

#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

/* Note: This is the pre-released op_api-version PTA for silu_backward. To avoid build problem, op_api namespace of
  defined function(NPUNativeOpApiFunctions) is removed here. When old-version PTA for silu_backward is refined and
  uploaded, the op_api namespace should be added (e.g. NPUNativeOpApiFunctions::silu_backward_out()), and the macro
  for compatibility(e.g. DO_COMPATIBILITY(aclnnSiluBackward, NPUNativeFunctions::silu_backward(grad_output, self)))
  should also be added in all defined functions.
*/ 

at::Tensor& silu_backward_out(const at::Tensor& grad_output, const at::Tensor& self,
                              at::Tensor& result) {
  OpPreparation::CheckOut({grad_output, self}, result, grad_output);
  EXEC_NPU_CMD(aclnnSiluBackward, grad_output, self, result);
  return result;
}

at::Tensor silu_backward(const at::Tensor& grad_output, const at::Tensor& self) {
  at::Tensor grad_input = OpPreparation::ApplyTensorWithoutFormat(grad_output);
  EXEC_NPU_CMD(aclnnSiluBackward, grad_output, self, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu
