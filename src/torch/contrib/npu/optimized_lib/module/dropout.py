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
import torch.nn as nn
import numpy as np


class DropoutV2(nn.Module):
    r"""Applies an NPU compatible dropout operation.

        This dropout method generates pseudo-random seed based on LCG(linear congruential generator) method.
        Since Ascend910 does not have a hardware unit that can generate real random numbers,
        we used the LCG method to generate pseudo-random seeds

        .. note::
            max_seed is a hyper-parameter strongly related to the underlying operator.
            Please check the MAX(2 ** 31 - 1 / 2 ** 10 - 1) in dropout_v2.py in the opp package for matching settings.
            By default, it is matched by the Pytorch and OPP packages.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = DropoutV2(p=0.5)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)
        """

    def __init__(self, p=0.5, inplace=False,
                 max_seed=2 ** 10 - 1):
        super(DropoutV2, self).__init__()

        self.p = p
        self.seed = torch.from_numpy(
            np.random.uniform(1, max_seed, size=(32 * 1024 * 12,)).astype(np.float32))

        self.checked = False

    def check_self(self, x):
        r"""Check device equipment between tensors.
        """
        if self.seed.device == x.device:
            self.checked = True
            return

        self.seed = self.seed.to(x.device)

    def forward(self, x):
        if not self.training:
            return x

        if not self.checked:
            self.check_self(x)

        x, mask, _ = torch.npu_dropoutV2(x, self.seed, p=self.p)
        return x


if __name__ == '__main__':
    torch.npu.set_device('npu:0')
    x = torch.randn(1, 2, 2, 2).npu()

    print('train mode')
    drop = DropoutV2()
    o = drop(x)
    print('input')
    print(x)
    print('output')
    print(o)

    print('eval mode')
    drop.eval()
    o = drop(x)
    print('input')
    print(x)
    print('output')
    print(o)


