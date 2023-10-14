import math
import torch
import torch.nn.functional as F

import torch_npu


def fuse_add_softmax_dropout(training, dropout, attn_mask, attn_scores, attn_head_size, p=0.5, dim=-1):
    """Using NPU custom operator to replace the native writing method to improve performance

    Examples::
    >>> training = True
    >>> dropout = nn.DropoutWithByteMask(0.1)
    >>> npu_input1 = torch.rand(96, 12, 384, 384).half().npu()
    >>> npu_input2 = torch.rand(96, 12, 384, 384).half().npu()
    >>> alpha = 0.125
    >>> axis = -1
    >>> output = fuse_add_softmax_dropout(training, dropout, npu_input1, npu_input2, alpha, p=axis)

        
    Args:
        training (bool): Whether it is training mode.
        dropout (nn.Module): the dropout layer
        attn_mask (Tensor): the attention mask
        attn_scores (Tensor): the raw attention scores
        attn_head_size (float): the head size
        p (float): probability of an element to be zeroed
        dim (int): A dimension along which softmax will be computed.

    Returns:
        torch.Tensor: The result of the mask operation
    """
    
    high_performance_support_hw = [128, 256, 384, 512]
    n, c, h, w = attn_scores.size()
    if h in high_performance_support_hw and (n * c) % 32 == 0:
        if training:
            drop_p = p
        else:
            drop_p = 0.
        _, _, attn_probs = torch_npu.npu_dropout_with_add_softmax(attn_scores, attn_mask,
                                                                1 / math.sqrt(attn_head_size), drop_p, -1)
    else:                  
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attn_scores = torch.add(attn_mask, attn_scores, alpha=(1 / math.sqrt(attn_head_size)))

        # Normalize the attention scores to probabilities.chrome
        attn_probs = F.softmax(attn_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attn_probs = dropout(attn_probs)    
    
    return attn_probs