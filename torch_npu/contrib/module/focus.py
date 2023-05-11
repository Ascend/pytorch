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
    _, _, w, _ = x.shape
    device = x.device
    if device.type == "npu":
        w_zerostart = torch.linspace(0, w - 1 - 1, w // 2, dtype=torch.float32, device=device).int()
        w_onestart = torch.linspace(1, w - 1, w // 2, dtype=torch.float32, device=device).int()
    else:
        w_zerostart = torch.linspace(0, w - 1 - 1, w // 2, dtype=torch.int64, device=device)
        w_onestart = torch.linspace(1, w - 1, w // 2, dtype=torch.int64, device=device)

    x_all_all_all_zerostart = x[..., ::2].contiguous()
    x_all_all_zerostart_zerostart = x_all_all_all_zerostart.index_select(2, w_zerostart)
    x_all_all_onestart_zerostart = x_all_all_all_zerostart.index_select(2, w_onestart)

    x_all_all_all_onestart = x[..., 1::2].contiguous()
    x_all_all_zerostart_onestart = x_all_all_all_onestart.index_select(2, w_zerostart)
    x_all_all_onestart_onestart = x_all_all_all_onestart.index_select(2, w_onestart)

    result = [x_all_all_zerostart_zerostart, x_all_all_onestart_zerostart, 
              x_all_all_zerostart_onestart, x_all_all_onestart_onestart]
    return result

class Focus(nn.Module):
    """Using NPU affinity writing method to replace the native Focus in Yolov5.
    
    Reference implementation link:
    https://github.com/ultralytics/yolov5/blob/4d05472d2b50108c0fcfe9208d32cb067a6e21b0/models/common.py#L227

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
