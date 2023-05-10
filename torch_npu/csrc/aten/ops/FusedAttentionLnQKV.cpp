// Copyright (c) 2022 Huawei Technologies Co., Ltd
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

#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_attention_ln_qkv_impl(
    at::Tensor& norm,
    at::Tensor& query_output,
    at::Tensor& key_output,
    at::Tensor& value_output,
    at::Tensor& mean,
    at::Tensor& variance,
    const at::Tensor& x,
    const at::Tensor& kernel_query,
    const at::Tensor& kernel_key,
    const at::Tensor& kernel_value,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    const at::Tensor& bias_query,
    const at::Tensor& bias_key,
    const at::Tensor& bias_value,
    double eps) {
  OpCommand cmd;
  cmd.Name("AttentionLnQKV")
      .Input(x)
      .Input(kernel_query)
      .Input(kernel_key)
      .Input(kernel_value)
      .Input(gamma)
      .Input(beta)
      .Input(bias_query)
      .Input(bias_key)
      .Input(bias_value)
      .Output(norm)
      .Output(query_output)
      .Output(key_output)
      .Output(value_output)
      .Output(mean)
      .Output(variance)
      .Attr("epsilon", static_cast<float>(eps))
      .Run();
  return std::tie(norm, query_output, key_output, value_output, mean, variance);
}

std::vector<at::Tensor> NPUNativeFunctions::npu_fused_attention_layernorm_qkv_fwd(
    const at::Tensor& x,
    const at::Tensor& kernel_query,
    const at::Tensor& kernel_key,
    const at::Tensor& kernel_value,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    const c10::optional<at::Tensor>& bias_query,
    const c10::optional<at::Tensor>& bias_key,
    const c10::optional<at::Tensor>& bias_value,
    int64_t seq_len,
    int64_t num_heads,
    double eps) {
  TORCH_CHECK(seq_len != 0 || num_heads != 0, 
      "seq_len and num_heads cannot be equal to 0.");
  c10::SmallVector<int64_t, SIZE> qkv_output_shape = {x.size(0) / seq_len, num_heads, seq_len, x.size(1) / num_heads};
  c10::SmallVector<int64_t, SIZE> mean_output_shape = {x.size(0)};
  at::Tensor norm = OpPreparation::ApplyTensor(x);
  at::Tensor query_output = OpPreparation::ApplyTensor(kernel_query, qkv_output_shape);
  at::Tensor key_output = OpPreparation::ApplyTensor(kernel_key, qkv_output_shape);
  at::Tensor value_output = OpPreparation::ApplyTensor(kernel_value, qkv_output_shape);
  at::Tensor mean = OpPreparation::ApplyTensorWithFormat(kernel_query, mean_output_shape, ACL_FORMAT_ND);
  at::Tensor variance = OpPreparation::ApplyTensorWithFormat(kernel_query, mean_output_shape, ACL_FORMAT_ND);

  const at::Tensor& bias_query_output = c10::value_or_else(bias_query, [] {return at::Tensor();});
  const at::Tensor& bias_key_output = c10::value_or_else(bias_key, [] {return at::Tensor();});
  const at::Tensor& bias_value_output = c10::value_or_else(bias_value, [] {return at::Tensor();});

  fused_attention_ln_qkv_impl(norm, query_output, key_output, value_output, mean, variance,
      x, kernel_query, kernel_key, kernel_value, gamma, beta, bias_query_output, bias_key_output, bias_value_output, eps);
  std::vector<at::Tensor> results = {norm, query_output, key_output, value_output, mean, variance};
  return results;
}

} // namespace native
} // namespace at_npu
