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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"
#include "c10/npu/SecondaryStreamGuard.h"
#include "c10/npu/NPUCachingAllocator.h"
#include <torch/csrc/autograd/record_function.h>

namespace at {
namespace native {
using namespace at::native::npu;

static const int64_t FZ_ALIGN_NUM = 16;
static const size_t BIAS_BUM = 4;

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> multi_head_attention_backward_npu(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& query_weight, const Tensor& key_weight, const Tensor& value_weight,
    const Tensor& out_proj_weight, const Tensor& query_bias, const Tensor& key_bias, const Tensor& value_bias,
    const Tensor& out_proj_bias, const Tensor& query_res, const Tensor& key_res, const Tensor& value_res,
    const Tensor& attn_scores, const Tensor& attn_res, const Tensor& context,
    const Tensor& y_grad, const Tensor& dropout_mask,
    int64_t attn_head_num, int64_t attn_dim_per_head,
    int64_t src_len, int64_t tgt_len,
    double dropout_prob, bool softmax_use_float
) {
    TORCH_CHECK(tgt_len > 0 && src_len > 0 && attn_head_num > 0 && attn_dim_per_head > 0,
        "tgt_len, src_len, attn_head_num, attn_dim_per_head should not equal zero.");
    TORCH_CHECK(tgt_len % FZ_ALIGN_NUM == 0 && src_len % FZ_ALIGN_NUM ==  0 &&
        attn_head_num % FZ_ALIGN_NUM ==  0 && attn_dim_per_head % FZ_ALIGN_NUM ==  0,
        "tgt_len, src_len, attn_head_num, attn_dim_per_head should align to 16.");
    auto query_shape = query.sizes();
    int64_t batch = query_shape[0] / tgt_len;
    auto weight_col = attn_head_num * attn_dim_per_head;

    Tensor query_weight_grad =  OpPreparation::ApplyTensor(query_weight, {weight_col, weight_col});
    Tensor key_weight_grad =  OpPreparation::ApplyTensor(key_weight, {weight_col, weight_col});
    Tensor value_weight_grad =  OpPreparation::ApplyTensor(value_weight, {weight_col, weight_col});
    Tensor out_proj_weight_grad =  OpPreparation::ApplyTensor(out_proj_weight, {weight_col, weight_col});
    Tensor query_grad =  OpPreparation::ApplyTensor(query, {query_shape[0], weight_col});
    Tensor key_grad =  OpPreparation::ApplyTensor(key, {batch * src_len, weight_col});
    Tensor value_grad =  OpPreparation::ApplyTensor(value, {batch * src_len, weight_col});
    Tensor query_bias_grad =  OpPreparation::ApplyTensor(query_bias, {1, weight_col});
    Tensor key_bias_grad =  OpPreparation::ApplyTensor(key_bias, {1, weight_col});
    Tensor value_bias_grad =  OpPreparation::ApplyTensor(value_bias, {1, weight_col});
    Tensor out_proj_bias_grad =  OpPreparation::ApplyTensor(out_proj_bias, {1, weight_col});

    vector<uint8_t> grad_mask(BIAS_BUM);
    grad_mask.clear();
    grad_mask.push_back(query_bias.defined());
    grad_mask.push_back(key_bias.defined());
    grad_mask.push_back(value_bias.defined());
    grad_mask.push_back(out_proj_bias.defined());
    at::ArrayRef<uint8_t> bias_grad_mask(grad_mask);

    OpCommand cmd;
    cmd.Name("MultiHeadAttentionGrad")
        .Input(query).Input(key).Input(value)
        .Input(query_weight).Input(key_weight).Input(value_weight)
        .Input(out_proj_weight).Input(query_res).Input(key_res).Input(value_res)
        .Input(attn_scores).Input(attn_res).Input(context).Input(y_grad);
    if (dropout_prob>0) {
        cmd.Input(dropout_mask);
    }
    cmd.Output(query_weight_grad).Output(key_weight_grad).Output(value_weight_grad).Output(out_proj_weight_grad)
        .Output(query_grad).Output(key_grad).Output(value_grad)
        .Output(query_bias_grad).Output(key_bias_grad).Output(value_bias_grad).Output(out_proj_bias_grad)
        .Attr("attn_head_num", attn_head_num).Attr("attn_dim_per_head", attn_dim_per_head)
        .Attr("src_len", src_len).Attr("tgt_len", tgt_len)
        .Attr("keep_prob", static_cast<float>(1 - dropout_prob)).Attr("softmax_use_float", softmax_use_float)
        .Attr("bias_grad_mask", bias_grad_mask)
        .Run();

    return std::tie(query_weight_grad, key_weight_grad, value_weight_grad, out_proj_weight_grad,
        query_grad, key_grad, value_grad, query_bias_grad, key_bias_grad, value_bias_grad, out_proj_bias_grad);
}
}} // namespace