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
using torch::autograd::AutogradContext;
using torch::autograd::Function;

class NPUFlashAttentionScoreFunction : public torch::autograd::Function<NPUFlashAttentionScoreFunction> {
public:
  static std::vector<at::Tensor> forward(
      AutogradContext *ctx, const at::Tensor &query_layer, const at::Tensor &key_layer,
      const at::Tensor &value_layer, const c10::optional<at::Tensor> &pse_opt,
      const c10::optional<at::Tensor> &drop_mask_opt, const c10::optional<at::Tensor> &padding_mask_opt,
      const c10::optional<at::Tensor> &atten_mask_opt, bool query_transpose, bool key_transpose, bool value_transpose,
      double scale, double keep_prob, bool is_transpose_out, int64_t pre_tockens, int64_t next_tockens)
  {
    const at::Tensor &pse = pse_opt.value_or(at::Tensor());
    const at::Tensor &padding_mask = padding_mask_opt.value_or(at::Tensor());
    const at::Tensor &atten_mask = atten_mask_opt.value_or(at::Tensor());
    const at::Tensor &drop_mask = drop_mask_opt.value_or(at::Tensor());
    at::Tensor attention_score;
    if (is_transpose_out) {
      attention_score = OpPreparation::ApplyTensor(query_layer,
          {query_layer.size(0), query_layer.size(2), query_layer.size(1) * query_layer.size(3)});
    } else {
      attention_score = OpPreparation::ApplyTensor(query_layer);
    }

    size_t dim = (query_layer.scalar_type() == at::ScalarType::Float) ? 8 : 16;
    at::Tensor softmax_max = OpPreparation::ApplyTensor(query_layer,
        {query_layer.size(0), query_layer.size(1), query_layer.size(2), dim});
    at::Tensor softmax_sum = OpPreparation::ApplyTensor(query_layer,
        {query_layer.size(0), query_layer.size(1), query_layer.size(2), dim});

    bool is_flash = true;
    at::Tensor softmax_out;
    EXEC_NPU_CMD_CLEAR_WORKSPACE(aclnnFlashAttentionScore, query_layer, key_layer, value_layer,
        pse_opt, drop_mask_opt, padding_mask_opt, atten_mask_opt,
        query_transpose, key_transpose, value_transpose,
        scale, keep_prob, is_transpose_out, pre_tockens, next_tockens, is_flash,
        softmax_max, softmax_sum, softmax_out, attention_score);

    at::AutoNonVariableTypeMode g;

    ctx->save_for_backward({query_layer, key_layer, value_layer, softmax_max, softmax_sum, softmax_out,
                            pse, drop_mask, padding_mask, atten_mask, attention_score});

    ctx->saved_data["is_flash"] = is_flash;
    ctx->saved_data["scale"] = scale;
    ctx->saved_data["keep_prob"] = keep_prob;
    ctx->saved_data["query_transpose"] = query_transpose;
    ctx->saved_data["key_transpose"] = key_transpose;
    ctx->saved_data["pre_tockens"] = pre_tockens;
    ctx->saved_data["next_tockens"] = next_tockens;
    ctx->saved_data["value_transpose"] = value_transpose;
    ctx->saved_data["is_transpose_out"] = is_transpose_out;

    return {attention_score};
  }

  static std::vector<at::Tensor> backward(AutogradContext *ctx, std::vector<at::Tensor> grad_outputs)
  {
    auto scale = ctx->saved_data["scale"].toDouble();
    auto keep_prob = ctx->saved_data["keep_prob"].toDouble();
    auto query_transpose = ctx->saved_data["query_transpose"].toBool();
    auto key_transpose = ctx->saved_data["key_transpose"].toBool();
    auto value_transpose = ctx->saved_data["value_transpose"].toBool();
    auto is_transpose_out = ctx->saved_data["is_transpose_out"].toBool();
    auto pre_tockens = ctx->saved_data["pre_tockens"].toInt();
    auto next_tockens = ctx->saved_data["next_tockens"].toInt();
    auto is_flash = ctx->saved_data["is_flash"].toBool();
    auto saved = ctx->get_saved_variables();

    auto query = saved[0];
    auto key = saved[1];
    auto value = saved[2];
    auto softmax_max = saved[3];
    auto softmax_sum = saved[4];
    auto softmax_out = saved[5];
    auto pse = saved[6];
    auto drop_mask = saved[7];
    auto padding_mask = saved[8];
    auto atten_mask = saved[9];
    auto attention_score = saved[10];

    bool dy_transpose = false;
    return NPUNativeOpApiFunctions::npu_flash_attention_score_grad(query,
        key, value, grad_outputs[0], pse, drop_mask, padding_mask, atten_mask,
        softmax_max, softmax_sum, softmax_out, attention_score, scale,
        keep_prob, query_transpose, key_transpose, value_transpose,
        dy_transpose, is_transpose_out, pre_tockens, next_tockens);
  }
};

std::vector<at::Tensor> NPUNativeOpApiFunctions::npu_flash_attention_score(
    const at::Tensor &query_layer, const at::Tensor &key_layer,
    const at::Tensor &value_layer, const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &drop_mask, const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask, bool query_transpose, bool key_transpose, bool value_transpose,
    double scale, double keep_prob, bool is_transpose_out, int64_t pre_tockens, int64_t next_tockens)
{
  return NPUFlashAttentionScoreFunction::apply(query_layer, key_layer, value_layer, pse, drop_mask, padding_mask,
      atten_mask, query_transpose, key_transpose, value_transpose, scale, keep_prob,
      is_transpose_out, pre_tockens, next_tockens);
}
} // namespace native
} // namespace at_npu
