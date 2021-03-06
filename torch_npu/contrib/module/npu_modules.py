# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
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

__all__ = ["DropoutWithByteMask"]


from torch.nn import Module
from ..function import npu_functional as F

class DropoutWithByteMask(Module):
    r"""Applies an NPU compatible DropoutWithByteMask operation, Only supports npu devices. 
    
    A new module for obtaining the performance benefits of operator fusion in graph mode.

    This DropoutWithByteMask method generates stateless random uint8 mask and do dropout according to the mask.

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

        >>> m = nn.DropoutWithByteMask(p=0.5)
        >>> input = torch.randn(16, 16)
        >>> output = m(input)
    """

    def __init__(self, p=0.5, inplace=False,
                 max_seed=2 ** 10 - 1):
        super(DropoutWithByteMask, self).__init__()

        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input1):
        return F.dropout_with_byte_mask(input1, self.p, self.training, self.inplace)
