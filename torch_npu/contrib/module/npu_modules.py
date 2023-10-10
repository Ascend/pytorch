__all__ = ["DropoutWithByteMask"]


from torch.nn import Module
from ..function import npu_functional as F


class DropoutWithByteMask(Module):
    r"""Applies an NPU compatible DropoutWithByteMask operation, Only supports npu devices. 
    
    A new module for obtaining the performance benefits of operator fusion in graph mode.

    This DropoutWithByteMask method generates stateless random uint8 mask and do dropout according to the mask.

    .. note::
        The performance is improved only in the device 32 core scenario.

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
