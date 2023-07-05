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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::_ctc_loss_backward(
    const at::Tensor& grad_out,
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
    const at::Tensor& neg_log_likelihood,
    const at::Tensor& log_alpha,
    int64_t blank,
    bool zeroInfinity) {
  DO_COMPATIBILITY(aclnnCtcLossBackward, NPUNativeFunctions::_ctc_loss_backward(grad_out, log_probs, targets, 
      input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zeroInfinity));
  
  auto outputSize = input_same_output_size(log_probs);

  // construct the output tensor of the NPU
  at::Tensor grad = OpPreparation::ApplyTensorWithoutFormat(grad_out, outputSize);

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnCtcLossBackward, grad_out, log_probs, targets, input_lengths, target_lengths, 
      neg_log_likelihood, log_alpha, blank, zeroInfinity, grad); 

  return grad;
}
} // namespace native
} // namespace at_npu
