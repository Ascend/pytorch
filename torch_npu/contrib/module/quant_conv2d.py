# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_2_t
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
import torch_npu

__all__ = ['QuantConv2d']


class QuantConv2d(nn.Module):

    r"""Applies Conv2d + Dequant: :math:`output = (fmap * weight + bias) * scale`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        output_dtype (str): Dtype of the output
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Padding added to all four sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        offset (bool, optional): If set to ``True``, the layer will learn an additive offset.
            Default: ``False``
        offset_x (int, optional): Actual padding value. Default: 0
        round_mode (str, optional): Requant calculation parameter. Default: "rint"
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor): the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        scale (Tensor): Dequant calculation parameter of shape (out_channels)
        offset (Tensor): Requant calculation parameter of shape (out_channels)

    Examples::
        >>> quant_conv2d_input = torch.randint(-1, 1, (1, 1, 4, 4), dtype=torch.int8)
        >>> weight = torch.randint(-1, 1, (1, 1, 3, 3), dtype=torch.int8)
        >>> scale = torch.randint(-1, 1, (1,), dtype=torch.int64)
        >>> bias = torch.randint(-1, 1, (1,), dtype=torch.int32)
        >>> model = QuantConv2d(in_channels, out_channels, k_size, output_dtype)
        >>> model = model.npu()
        >>> model.weight.data = weight
        >>> model.scale.data = scale
        >>> model.bias.data = bias
        >>> config = CompilerConfig()
        >>> npu_backend = tng.get_npu_backend(compiler_config=config)
        >>> static_graph_model = torch.compile(model, backend=npu_backend, dynamic=False)
        >>> output = static_graph_model(quant_conv2d_input)
        >>> print(output.size())
        torch.Size(1, 1, 2, 2)
    """
    in_channels: int
    out_channels: int
    k_size: int
    output_dtype: str

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 output_dtype: torch.dtype,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 offset: bool = False,
                 offset_x: int = 0,
                 round_mode: str = "rint",
                 device=None,
                 dtype=None) -> None:
        super(QuantConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.offset_x = offset_x
        self.round_mode = round_mode
        self.output_dtype = output_dtype

        self.weight = \
            Parameter(torch.empty((self.out_channels, self.in_channels, *self.kernel_size), dtype=torch.int8), False)
        self.scale = Parameter(torch.empty(self.out_channels, dtype=torch.int64), False)
        if bias:
            self.bias = Parameter(torch.empty(self.out_channels, dtype=torch.int32), False)
        else:
            self.register_parameter('bias', None)
        if offset:
            self.offset = Parameter(torch.empty(out_channels, dtype=torch.float32), False)
        else:
            self.register_parameter('offset', None)

    def forward(self, quant_conv2d_input: Tensor) -> Tensor:
        scale_ = self.scale
        if self.scale.dtype == torch.float32:
            scale_ = torch_npu.npu_trans_quant_param(self.scale)
        return torch_npu.npu_quant_conv2d(quant_conv2d_input, self.weight, scale_, self.stride, self.padding,
                                          self.dilation, self.groups, self.offset_x, self.round_mode,
                                          self.output_dtype, self.bias, self.offset)
