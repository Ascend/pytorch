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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

std::vector<at::Tensor> NPUNativeOpApiFunctions::npu_flash_attention_score_grad(
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

  at::Tensor dq, dk, dv;
  dq = OpPreparation::ApplyTensor(query);
  dk = OpPreparation::ApplyTensor(key);
  dv = OpPreparation::ApplyTensor(value);

  EXEC_NPU_CMD(aclnnFlashAttentionScoreGrad, query, key, value, dy, pse, drop_mask, padding_mask, atten_mask,
               softmax_max, softmax_sum, softmax_in, attention_in, scale_value, keep_prob, query_transpose,
               key_transpose, value_transpose, dy_transpose, is_transpose_attention, pre_tockens, next_tockens,
               is_flash, dq, dk, dv);

  return {dq, dk, dv,
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
}

} // namespace native
} // namespace at_npu
