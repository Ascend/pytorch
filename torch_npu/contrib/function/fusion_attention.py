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


def npu_fusion_attention(query, key, value, head_num, input_layout, pse=None, padding_mask=None, atten_mask=None,
                         scale=1., keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647, 
                         inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                         gen_mask_parallel=True, sync=False):
    return torch_npu.npu_flash_attention(
        query, key, value, head_num, input_layout, pse=pse, padding_mask=padding_mask, atten_mask=atten_mask,
        scale=scale, keep_prob=keep_prob, pre_tockens=pre_tockens, next_tockens=next_tockens,
        inner_precise=inner_precise, prefix=prefix, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode, gen_mask_parallel=gen_mask_parallel, sync=sync)


def npu_fusion_attention_grad(query, key, value, dy, head_num, input_layout, pse=None, padding_mask=None,
                              atten_mask=None, softmax_max=None, softmax_sum=None, softmax_in=None, attention_in=None,
                              scale_value=1., keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647,
                              inner_precise=0, seed=0, offset=0, numels=0, prefix=None, actual_seq_qlen=None,
                              actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False):
    return torch_npu.npu_flash_attention_grad(
        query, key, value, dy, head_num, input_layout, pse=pse, padding_mask=padding_mask, atten_mask=atten_mask,
        softmax_max=softmax_max, softmax_sum=softmax_sum, softmax_in=softmax_in, attention_in=attention_in,
        scale_value=scale_value, keep_prob=keep_prob, pre_tockens=pre_tockens, next_tockens=next_tockens,
        inner_precise=inner_precise, seed=seed, offset=offset, numels=numels, prefix=prefix,
        actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen, sparse_mode=sparse_mode,
        gen_mask_parallel=gen_mask_parallel, sync=sync)
