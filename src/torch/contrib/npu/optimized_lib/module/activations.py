# Copyright (c) 2021, Huawei Technologies.All rights reserved.
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
import torch.nn as nn

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
        x = torch.npu_mish(x)
        return x


if __name__ == '__main__':
    torch.npu.set_device('npu:0')
    x = torch.randn(2, 32, 4, 4)
    x.requires_grad = True
    model = Mish()

    x = x.npu()
    model = model.npu()

    o = model(x)
    l = o.sum()
    l.backward()

    o = model(x.half())
    l = o.sum()
    l.backward()

    torch.npu.synchronize()
    print('Mish test success.')
