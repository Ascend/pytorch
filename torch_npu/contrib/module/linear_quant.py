import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch_npu
from torch_npu.utils._error_code import ErrCode, ops_error

__all__ = ["LinearQuant"]


class LinearQuant(nn.Module):
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
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        scale: quant matmul calculation parameter
        offset: quant matmul calculation parameter
        pertoken_scale: inverse quant matmul calculation parameter
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    A4W4 Examples::
        >>> x1 = torch.randint(-1, 1, (1, 2), dtype=torch.int32).npu()
        >>> x2 = torch.randint(-1, 1, (128, 2), dtype=torch.int32).npu()
        >>> scale = torch.randn(1, dtype=torch.float32).npu()
        >>> model = LinearQuant(in_features, out_features, False)
        >>> model = model.npu()
        >>> model.weight.data = x2
        >>> model.scale.data = scale
        >>> output = model(x1)
        >>> print(output.size())
        torch.Size(1, 128)

    A8W8 Examples::
        >>> x1 = torch.randint(-1, 1, (1, 5), dtype=torch.int8).npu()
        >>> x2 = torch.randint(-1, 1, (127, 5), dtype=torch.int8).npu()
        >>> scale = torch.randn(1, dtype=torch.float32).npu()
        >>> model = LinearQuant(in_features, out_features, False)
        >>> model = model.npu()
        >>> model.weight.data = x2
        >>> model.scale.data = scale
        >>> output = model(x1)
        >>> print(output.size())
        torch.Size(1, 127)
    """
    in_features: int
    out_features: int
    weight: Tensor
    scale: Tensor
    offset: Tensor
    pertoken_scale: Tensor
    bias: Tensor

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, offset: bool = False,
                 pertoken_scale: bool = False, device=None, dtype=None, output_dtype=None) -> None:

        super(LinearQuant, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features)), False)
        self.scale = Parameter(torch.empty(out_features), False)
        self.output_dtype = output_dtype
        if offset:
            self.offset = Parameter(torch.empty(out_features, dtype=torch.float32), False)
        else:
            self.register_parameter('offset', None)

        if pertoken_scale:
            self.pertoken_scale = Parameter(torch.empty(out_features, dtype=torch.float32), False)
        else:
            self.register_parameter('pertoken_scale', None)

        if bias:
            self.bias = Parameter(torch.empty(out_features, dtype=torch.int32), False)
        else:
            self.register_parameter('bias', None)

    def forward(self, linear_quant_input: Tensor) -> Tensor:
        scale_quant = self.scale
        first_last_dim = self.weight.dim() - 1
        second_last_dim = self.weight.dim() - 2
        if not ((linear_quant_input.dtype == torch.int32 and self.weight.dtype == torch.int32) or
                (linear_quant_input.dtype == torch.int8 and self.weight.dtype == torch.int8)):
            raise ValueError("input and weight should be both torch.int32 or both torch.int8 datatype, "
                             f"but now input is {linear_quant_input.dtype}, weight is {self.weight.dtype}." + ops_error(ErrCode.TYPE))

        is_check_dtype_ok = (self.scale.dtype == torch.float32 and 
                             self.output_dtype not in [torch.bfloat16, torch.int32])
        if self.pertoken_scale is None and is_check_dtype_ok:
            scale_quant = torch_npu.npu_trans_quant_param(self.scale, self.offset)

        return torch_npu.npu_quant_matmul(linear_quant_input, self.weight.transpose(second_last_dim, first_last_dim),
                                          scale_quant, offset=self.offset, pertoken_scale=self.pertoken_scale, bias=self.bias,
                                          output_dtype=self.output_dtype)
