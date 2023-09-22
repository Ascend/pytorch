# Copyright (c) 2023, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch_npu


def npu_fusion_attention(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask,
                         scale=1., keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0):
    return torch_npu.npu_flash_attention(
        query, key, value, head_num, input_layout, pse, padding_mask, atten_mask,
        scale, keep_prob, pre_tockens, next_tockens, inner_precise=inner_precise)


def npu_fusion_attention_grad(query, key, value, dy, head_num, input_layout, pse, padding_mask, atten_mask,
                              softmax_max, softmax_sum, softmax_in, attention_in, scale_value=1., keep_prob=1.,
                              pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0):
    return torch_npu.npu_flash_attention_grad(
        query, key, value, dy, head_num, input_layout, pse, padding_mask, atten_mask,
        softmax_max, softmax_sum, softmax_in, attention_in, scale_value, keep_prob,
        pre_tockens, next_tockens, inner_precise=inner_precise)
