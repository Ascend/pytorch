# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

import torch
import torch_npu

__all__ = ["MatmulApply"]

class MatmulApply(torch.autograd.Function):
    """Using NPU custom operator to replace the native writing method to improve performance.
    
    Compute Function:
    attn = (q @ k.transpose(-2, -1))
    
    Reference implementation link:
    https://github.com/huggingface/transformers/blob/d6eeb871706db0d64ab9ffd79f9545d95286b536/src/transformers/models/bert/modeling_bert.py#L331

    This interface is faster than the original on NPU.

    .. note::
        In the dynamic shape scene, Due to the operator restriction, the broadcast scene is not supported.

    Args:
        tensor1 (Tensor): the first tensor to be multiplied.
        tensor2 (Tensor): the second tensor to be multiplied.

    Returns:
        Tensor: the output tensor.

    Examples::
        >>> tensor1 = torch.randn(68, 5, 75, 16).npu()
        >>> tensor1.requires_grad_(True)
        >>> tensor2 = torch.randn(68, 5, 75, 16).npu()
        >>> tensor2.requires_grad_(True)
        >>> output = matmul_transpose(tensor1, tensor2)
        >>> output.sum().backward()
    """
    @staticmethod
    def forward(ctx, self, mat2):
        ctx.save_for_backward(self, mat2)
        result = torch.matmul(self, mat2.transpose(-2, -1))
        return result.detach()

    @staticmethod
    def backward(ctx, grad):
        self, mat2 = ctx.saved_tensors
        self_grad = torch.npu_bmmV2(grad, mat2, [])
        mat2_grad = torch.npu_bmmV2(grad.transpose(-2, -1), self, [])
        return self_grad, mat2_grad

matmul_transpose = MatmulApply.apply
