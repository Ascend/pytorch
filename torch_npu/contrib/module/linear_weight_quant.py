import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch_npu

__all__ = ["LinearWeightQuant"]


class LinearWeightQuant(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        offset: If set to ``True``, the layer will learn an additive offset.
            Default: ``False``
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
        antiquant_scale: weight quant batchmatmul calculation parameter
        antiquant_offset: weight quant batchmatmul calculation parameter
        quant_scale: weight quant batchmatmul calculation parameter
        quant_offset: weight quant batchmatmul calculation parameter
        bias: the learnable bias of the module of shape
        antiquant_group_size: size of group in antiquant calculation

    Examples::
        >>> x = torch.randn((16, 32), dtype=torch.float16).npu()
        >>> weight = torch.randint(-3, 3, (128, 32), dtype=torch.int8).npu()
        >>> antiquant_scale = torch.randn((128), dtype=torch.float16).npu()
        >>> model = LinearWeightQuant(32, 128, False)
        >>> model = model.npu()
        >>> model.weight.data = weight
        >>> model.antiquant_scale.data = antiquant_scale
        >>> output = model(x)
        >>> print(output.size())
        torch.Size(16, 128)
    """
    in_features: int
    out_features: int
    weight: Tensor
    antiquant_scale: Tensor
    antiquant_offset: Tensor
    quant_scale: Tensor
    quant_offset: Tensor

    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        device=None,
        dtype=None,
        antiquant_offset: bool = False,
        quant_scale: bool = False,
        quant_offset: bool = False,
        antiquant_group_size: int = 0,
        inner_precise: int = 0,
    ) -> None:
        super(LinearWeightQuant, self).__init__()
        self.weight = Parameter(torch.empty((out_features, in_features), device=device), False)
        self.antiquant_scale = Parameter(torch.empty(out_features, device=device), False)

        if antiquant_offset:
            self.antiquant_offset = Parameter(torch.empty(out_features, device=device), False)
        else:
            self.register_parameter('antiquant_offset', None)

        if quant_scale:
            self.quant_scale = Parameter(torch.empty(out_features, device=device), False)
        else:
            self.register_parameter('quant_scale', None)

        if quant_offset:
            self.quant_offset = Parameter(torch.empty(out_features, device=device), False)
        else:
            self.register_parameter('quant_offset', None)

        if bias:
            self.bias = Parameter(torch.empty(out_features, device=device), False)
        else:
            self.register_parameter('bias', None)

        self.antiquant_group_size = antiquant_group_size
        self.inner_precise = inner_precise

    def forward(self, x: Tensor) -> Tensor:
        antiquant_scale = self.antiquant_scale
        antiquant_offset = self.antiquant_offset
        if self.antiquant_scale.dim() == 2:
            antiquant_scale = self.antiquant_scale.transpose(-1, -2)
        if self.antiquant_offset is not None:
            if self.antiquant_offset.dim() == 2:
                antiquant_offset = self.antiquant_offset.transpose(-1, -2)
        return torch_npu.npu_weight_quant_batchmatmul(x, self.weight.transpose(-1, -2), antiquant_scale,
                                                      antiquant_offset, self.quant_scale, self.quant_offset,
                                                      self.bias, self.antiquant_group_size, self.inner_precise)
