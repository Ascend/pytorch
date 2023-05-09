import torch
import torch.nn as nn
import torch_npu

class Mish(nn.Module):
    def __init__(self):
        r"""Applies an NPU based Mish operation.

        Origin CUDA implement link:
        https://github.com/thomasbrandon/mish-cuda

        Paper link:
        [Mish: A Self Regularized Non-Monotonic Activation Function]
        (https://www.bmvc2020-conference.com/assets/papers/0928.pdf)

        Official implementation based on PyTorch link:
        https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py

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

    def forward(self, x):
        x = torch_npu.npu_mish(x)
        return x

class SiLU(nn.Module):
    def __init__(self):
        r"""Applies an NPU based Sigmoid Linear Unit (SiLU) function, element-wise.
        The SiLU function is also known as the swish function.

        .. math::
            \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

        .. note::
            See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
            where the SiLU (Sigmoid Linear Unit) was originally coined, and see
            `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
            in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
            a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
            where the SiLU was experimented with later.

            SiLU exists in the official version since PyTorch 1.7.0.
            Currently, the PyTorch version adapted for NPU is 1.5.0,
            so SiLU needs to be defined as an additional module.

        Examples::
            >>> m = nnn.SiLU()
            >>> input_tensor = torch.randn(2, 32, 5, 5)
            >>> output = m(input_tensor)
        """
        super(SiLU, self).__init__()

    def forward(self, x):
        x = torch_npu.npu_silu(x)
        return x

Swish = SiLU