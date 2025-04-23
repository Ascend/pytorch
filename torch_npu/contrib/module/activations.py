import warnings

import torch
import torch.nn as nn
import torch_npu

warnings.filterwarnings(action='once', category=FutureWarning)


class Mish(nn.Module):
    def __init__(self):
        r"""Applies an NPU based Mish operation.

        The calculation formula is as follows:
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

        .. note::
            Mish exists in the official version  in PyTorch 1.9.0.
            Currently, the PyTorch version adapted for NPU is 1.5.0,
            so Mish needs to be defined as an additional module.

        Examples::
            >>> m = nnn.Mish()
            >>> input_tensor = torch.randn(2, 32, 5, 5)
            >>> output = m(input_tensor)
        """
        super(Mish, self).__init__()
        
        warnings.warn("torch_npu.contrib.module.Mish is deprecated. "
                      "Please use torch.nn.Mish for replacement.", FutureWarning)

    def forward(self, x):
        x = torch_npu.npu_mish(x)
        return x


class SiLU(nn.Module):
    def __init__(self):
        r"""Applies an NPU based Sigmoid Linear Unit (SiLU) function, element-wise.
        The SiLU function is also known as the swish function.

        .. math::
            \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

        Examples::
            >>> m = nnn.SiLU()
            >>> input_tensor = torch.randn(2, 32, 5, 5)
            >>> output = m(input_tensor)
        """
        super(SiLU, self).__init__()
        
        warnings.warn("torch_npu.contrib.module.SiLU is deprecated. "
                      "Please use torch.nn.SiLU for replacement.", FutureWarning)

    def forward(self, x):
        x = torch_npu.npu_silu(x)
        return x


Swish = SiLU
