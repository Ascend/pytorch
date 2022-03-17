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

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> multi_head_attention_npu(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& query_weight, const Tensor& key_weight, const Tensor& value_weight,
    const Tensor& attn_mask, const Tensor& out_proj_weight,
    const Tensor& query_bias, const Tensor& key_bias, const Tensor& value_bias,
    const Tensor& out_proj_bias, const Tensor& mask,
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
    
    auto query_options = query.options();
    auto query_format = CalcuOpUtil::get_tensor_npu_format(query);

    Tensor y = at::empty_with_format(
        {query_shape[0], weight_col}, query_options, query_format);
    Tensor dropout_mask = at::empty_with_format(
        {batch * attn_head_num * tgt_len * src_len / 8}, query.options().dtype(kByte), ACL_FORMAT_ND);
    Tensor query_res =  at::empty_with_format(
        {batch, attn_head_num, tgt_len, attn_dim_per_head}, query_options, query_format);
    Tensor key_res = at::empty_with_format(
        {batch, attn_head_num, src_len, attn_dim_per_head}, query_options, query_format);
    Tensor value_res = at::empty_with_format(
        {batch, attn_head_num, src_len, attn_dim_per_head}, query_options, query_format);
    Tensor attn_scores;
    if (softmax_use_float) {
        attn_scores = at::empty_with_format(
            {batch, attn_head_num, tgt_len, src_len}, query.options().dtype(kFloat), query_format);
    } else {
        attn_scores = at::empty_with_format(
            {batch, attn_head_num, tgt_len, src_len}, query_options, query_format);
    }
    Tensor attn_res = at::empty_with_format(
        {batch, attn_head_num, tgt_len, src_len}, query_options, query_format);
    Tensor context =  at::empty_with_format(
        {query_shape[0], weight_col}, query_options, query_format);

    OpCommand cmd;
    cmd.Name("MultiHeadAttention")
        .Input(query).Input(key).Input(value)
        .Input(query_weight).Input(key_weight).Input(value_weight)
        .Input(attn_mask).Input(out_proj_weight);
    if (query_bias.defined()) {
        cmd.Input(query_bias);
    }
    if (key_bias.defined()) {
        cmd.Input(key_bias);
    }
    if (value_bias.defined()) {
        cmd.Input(value_bias);
    }
    if (out_proj_bias.defined()) {
        cmd.Input(out_proj_bias);
    }
    if (mask.defined()) {
        cmd.Input(mask);
    }
    cmd.Output(y).Output(dropout_mask).Output(query_res).Output(key_res).Output(value_res)
        .Output(attn_scores).Output(attn_res).Output(context)
        .Attr("attn_head_num", attn_head_num).Attr("attn_dim_per_head", attn_dim_per_head)
        .Attr("src_len", src_len).Attr("tgt_len", tgt_len)
        .Attr("keep_prob", static_cast<float>(1 - dropout_prob)).Attr("softmax_use_float", softmax_use_float)
        .Run();

    return std::tie(y, dropout_mask, query_res, key_res, value_res, attn_scores, attn_res, context);
}
}} // namespace at::native