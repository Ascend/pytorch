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

#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor format_trans(const at::Tensor &at_tensor) {
    return at_tensor.defined() ? NPUNativeFunctions::npu_format_cast(at_tensor, ACL_FORMAT_ND) : at_tensor;
}
std::vector<at::Tensor> NPUNativeFunctions::npu_flash_attention_grad(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &dy,
    const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &drop_mask,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_max,
    const c10::optional<at::Tensor> &softmax_sum,
    const c10::optional<at::Tensor> &softmax_in,
    const c10::optional<at::Tensor> &attention_in,
    double scale_value,
    double keep_prob,
    bool query_transpose,
    bool key_transpose,
    bool value_transpose,
    bool dy_transpose,
    bool is_transpose_attention,
    int64_t pre_tockens,
    int64_t next_tockens)
{
  bool is_flash = true;
  const at::Tensor &pse_const = pse.value_or(at::Tensor());
  const at::Tensor &drop_mask_const = drop_mask.value_or(at::Tensor());
  const at::Tensor &padding_mask_const = padding_mask.value_or(at::Tensor());
  const at::Tensor &atten_mask_const = atten_mask.value_or(at::Tensor());
  const at::Tensor &softmax_max_const = softmax_max.value_or(at::Tensor());
  const at::Tensor &softmax_sum_const = softmax_sum.value_or(at::Tensor());
  const at::Tensor &softmax_const = softmax_in.value_or(at::Tensor());
  const at::Tensor &attention_const = attention_in.value_or(at::Tensor());

  at::Tensor format_query = NPUNativeFunctions::npu_format_cast(query, ACL_FORMAT_ND);
  at::Tensor format_key = NPUNativeFunctions::npu_format_cast(key, ACL_FORMAT_ND);
  at::Tensor format_value = NPUNativeFunctions::npu_format_cast(value, ACL_FORMAT_ND);
  at::Tensor format_dy = NPUNativeFunctions::npu_format_cast(dy, ACL_FORMAT_ND);

  at::Tensor format_pse = format_trans(pse_const);
  at::Tensor format_drop_mask = format_trans(drop_mask_const);
  at::Tensor format_padding_mask = format_trans(padding_mask_const);
  at::Tensor format_atten_mask = format_trans(atten_mask_const);
  at::Tensor format_softmax_max = format_trans(softmax_max_const);
  at::Tensor format_softmax_sum = format_trans(softmax_sum_const);
  at::Tensor format_softmax = format_trans(softmax_const);
  at::Tensor format_attention = format_trans(attention_const);

  at::Tensor dq, dk, dv;
  dq = OpPreparation::ApplyTensor(format_query);
  dk = OpPreparation::ApplyTensor(format_key);
  dv = OpPreparation::ApplyTensor(format_value);

  EXEC_NPU_CMD_CLEAR_WORKSPACE(
      aclnnFlashAttentionScoreGrad, format_query, format_key, format_value, format_dy,
      format_pse, format_drop_mask, format_padding_mask, format_atten_mask,
      format_softmax_max, format_softmax_sum, format_softmax, format_attention, scale_value, keep_prob, query_transpose,
      key_transpose, value_transpose, dy_transpose, is_transpose_attention, pre_tockens, next_tockens,
      is_flash, dq, dk, dv);

  return {dq, dk, dv,
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
}

} // namespace native
} // namespace at_npu
