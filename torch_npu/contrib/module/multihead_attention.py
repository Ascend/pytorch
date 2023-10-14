from typing import Dict, Optional, Tuple, List
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import torch_npu
from torch_npu.contrib.module.ensemble_dropout import NpuCachedDropout, DropOutTask
from ..function import matmul_transpose

dropout_class = NpuCachedDropout


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    if not isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
        raise TypeError("Expected isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))")

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        if module.weight.size(1) % block_size != 0:
            raise ValueError("Input features must be a multiple of block sizes")

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            if module.in_channels % block_size != 0:
                raise ValueError("Input channels must be a multiple of block sizes")
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            if k % block_size != 0:
                raise ValueError("Kernel size must be a multiple of block size")

    def _forward_pre_hook(mod, input1):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


class NpuLinear(nn.Linear):
    def forward(self, input2):
        input_shape = input2.size()
        if input2.dim() == 3:
            input2 = input2.view(-1, self.in_features)
            return torch_npu.npu_linear(input2, self.weight, self.bias).view(
                 input_shape[0], input_shape[1], self.out_features)
        elif input2.dim() == 2:
            return torch_npu.npu_linear(input2, self.weight, self.bias)
        else:
            raise RuntimeError('not support this dim')

        
class MHAConfig:
    use_fussion_mha = False

    @classmethod
    def set_fussion(cls):
        from torch_npu import npu_multi_head_attention
        cls.use_fussion_mha = True


def Matmul_transpose(tensor1, tensor2):
    return matmul_transpose.MatmulApply.apply(tensor1, tensor2)


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.

    .. note::
        Dynamic shapes are not supported.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads. 
        kdim(int): Total number of features for keys. Default: None
        vdim(int): Total number of features for values. Default: None
        dropout (float): Dropout probability 
        bias (bool):  If specified, adds bias to input / output projection layers. Default: True.
        add_bias_kv (bool): If specified, adds bias to the key and value sequences at dim=0. Default: False.
        add_zero_attn (bool): If specified, adds a new batch of zeros to the key and value sequences at dim=1. 
                              Default: False.
        self_attention (bool): Calculate your own attention score. Default: False.
        encoder_decoder_attention (bool): The input is the output of the encoder and the self-attention 
                                          output of the decoder, where the self-attention of the encoder 
                                          is used as the key and value, and the self-attention of the decoder 
                                          is used as the query. Default: False.
        q_noise(float): amount of Quantization Noise
        qn_block_size(int): size of the blocks for subsequent quantization with iPQ
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = dropout_class(
            dropout, module_name=self.__class__.__name__
        )
        self.dropout_prob = dropout

        self.use_dropout_optim = (dropout_class is NpuCachedDropout)

        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        if self.self_attention and not self.qkv_same_dim:
            raise ValueError("Self-attention requires query, key and " "value to be of the same size")

        self.k_proj = quant_noise(
            NpuLinear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            NpuLinear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            NpuLinear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            NpuLinear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def transpose_for_scores(self, x):
        new_x_shape = (self.batch_size, self.squence_length) + (self.num_attention_heads, self.attention_head_size)
        return torch_npu.npu_confusion_transpose(x, (0, 2, 1, 3), new_x_shape, False)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor], bsz, tgt_len, s_len,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if MHAConfig.use_fussion_mha:
            attn = self.multi_attn(query, key, value, key_padding_mask, bsz, tgt_len)
            return attn, None
        else:
            return None, None

    def multi_attn(self, query, key, value, key_padding_mask, bsz, tgt_len):
        src_len = key.size(0) // bsz
        if self.use_dropout_optim:
            dropout_mask = self.dropout_module([(bsz, self.num_heads, tgt_len, src_len), query.dtype, query.device])
        else:
            dropout_mask = None
        attn = torch_npu.npu_multi_head_attention(query, key, value, self.q_proj.weight,
                                                 self.k_proj.weight, self.v_proj.weight,
                                                 key_padding_mask, self.out_proj.weight,
                                                 self.q_proj.bias, self.k_proj.bias, self.v_proj.bias,
                                                 self.out_proj.bias, dropout_mask, self.num_heads,
                                                 self.head_dim, src_len, tgt_len, self.dropout_prob, True)
        return attn[0]

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.half(), key_padding_mask.half()], dim=3
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, key_padding_mask.size(1), key_padding_mask.size(2),
                 src_len - prev_key_padding_mask.size(3)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.half(), filler.half()], dim=3
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, key_padding_mask.size(1), key_padding_mask.size(2),
                 src_len - key_padding_mask.size(3)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.half(), key_padding_mask.half()], dim=3
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
