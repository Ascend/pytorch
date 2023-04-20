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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::_ctc_loss_backward(
    const at::Tensor& grad_out,
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
    const at::Tensor& neg_log_likelihood,
    const at::Tensor& log_alpha,
    int64_t blank,
    bool zeroInfinity) {
  at::Tensor grad_out_cast = grad_out;
  if (grad_out.scalar_type() == at::ScalarType::Half) {
    grad_out_cast = NPUNativeFunctions::npu_dtype_cast(grad_out, at::ScalarType::Float);
  }

  at::Tensor log_probs_cast = log_probs;
  if (log_probs.scalar_type() == at::ScalarType::Half) {
    log_probs_cast = NPUNativeFunctions::npu_dtype_cast(log_probs, at::ScalarType::Float);
  }

  at::Tensor neg_log_likelihood_cast = neg_log_likelihood;
  if (neg_log_likelihood.scalar_type() == at::ScalarType::Half) {
    neg_log_likelihood_cast = NPUNativeFunctions::npu_dtype_cast(neg_log_likelihood, at::ScalarType::Float);
  }

  at::Tensor log_alpha_cast = log_alpha;
  if (log_alpha.scalar_type() == at::ScalarType::Half) {
    log_alpha_cast = NPUNativeFunctions::npu_dtype_cast(log_alpha, at::ScalarType::Float);
  }

  auto input_lengths_tensor = at::tensor(input_lengths, targets.options().dtype(at::kInt));
  auto target_lengths_tensor = at::tensor(target_lengths, targets.options().dtype(at::kInt));

  auto outputSize = input_same_output_size(log_probs);

  // construct the output tensor of the NPU
  at::Tensor grad = OpPreparation::ApplyTensor(log_probs_cast, outputSize);
  // calculate the output result of the NPU
  OpCommand cmd;
  cmd.Name("CTCLossV2Grad")
      .Input(grad_out_cast)
      .Input(log_probs_cast)
      .Input(targets)
      .Input(input_lengths_tensor)
      .Input(target_lengths_tensor)
      .Input(neg_log_likelihood_cast)
      .Input(log_alpha_cast)
      .Output(grad)
      .Attr("blank", blank)
      .Attr("zero_infinity", zeroInfinity)
      .Run();

  if (grad_out.scalar_type() == at::ScalarType::Half) {
    grad = NPUNativeFunctions::npu_dtype_cast(grad, at::ScalarType::Half);
  }
  
  return grad;
}
} // namespace native
} // namespace at_npu
