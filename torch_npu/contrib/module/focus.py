import torch
import torch.nn as nn
import torch_npu


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def fast_slice(x):
    return [x[..., ::2, ::2], x[..., 1::2, ::2],
            x[..., ::2, ::2], x[..., 1::2, 1::2]]


class Focus(nn.Module):
    """Using NPU affinity writing method to replace the native Focus in Yolov5.

    Args:
        c1 (int): Number of channels in the input image.
        c2 (int): Number of channels produced by the convolution. 
        k(int): Size of the convolving kernel. Default: 1
        s(int): Stride of the convolution. Default: 1
        p (int): padding 
        g (int):  Number of blocked connections from input channels to output channels. Default: 1
        act (bool): whether to use an activation function. Default: True.

    Examples::
        >>> input = torch.randn(4, 8, 300, 40).npu()
        >>> input.requires_grad_(True)
        >>> fast_focus = Focus(8, 13).npu()
        >>> output = fast_focus(input)
        >>> output.sum().backward()
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat(fast_slice(x), 1))
