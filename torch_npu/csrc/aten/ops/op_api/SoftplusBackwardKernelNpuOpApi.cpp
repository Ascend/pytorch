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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::softplus_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold,
    at::Tensor& grad_input) {
  DO_COMPATIBILITY(aclnnSoftplusBackward, NPUNativeFunctions::softplus_backward_out(grad_output, self, beta, 
                                                                                    threshold, grad_input));
  EXEC_NPU_CMD(aclnnSoftplusBackward, grad_output, self, beta, threshold, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu
