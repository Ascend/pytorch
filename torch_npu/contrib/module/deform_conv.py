import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair, _single
import torch_npu


class ModulatedDeformConv2dFunction(Function):

    @staticmethod
    def forward(ctx,
                input_tensor,
                offset_ori,
                mask,
                weight,
                bias=None,
                with_bias=False,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deformable_groups=1,
                sort_index_for_npu_fp=None,
                sort_index_for_npu_bp=None,
                ):

        input_tensor = input_tensor.float()
        offset_ori = offset_ori.float()
        mask = mask.float()

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.sort_index_for_npu_bp = sort_index_for_npu_bp
        ctx.with_bias = with_bias

        offset = offset_ori.index_select(1, sort_index_for_npu_fp)
        offset_all = torch.cat([offset, mask], dim=1)
        output, offset_out = torch_npu.npu_deformable_conv2d(
            input_tensor, weight, offset_all, bias,
            kernel_size=[weight.shape[3], weight.shape[2]],
            stride=[1, 1, ctx.stride, ctx.stride],
            padding=[ctx.padding, ctx.padding, ctx.padding, ctx.padding],
            dilation=[1, 1, ctx.dilation, ctx.dilation],
            groups=ctx.groups, deformable_groups=ctx.deformable_groups,
            modulated=True)
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or input_tensor.requires_grad:
            ctx.save_for_backward(input_tensor, weight, offset_out, offset_all)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight, offset_out, offset_all = ctx.saved_tensors
        grad_input, grad_weight, grad_offset_all, grad_bias = torch_npu.npu_deformable_conv2dbk(
            input_tensor, grad_output, offset_out, weight, offset_all,
            kernel_size=[weight.shape[3], weight.shape[2]],
            stride=[1, 1, ctx.stride, ctx.stride],
            padding=[ctx.padding, ctx.padding, ctx.padding, ctx.padding],
            dilation=[1, 1, ctx.dilation, ctx.dilation],
            groups=ctx.groups, deformable_groups=ctx.deformable_groups, modulated=True)
        grad_offset = grad_offset_all.index_select(1, ctx.sort_index_for_npu_bp)
        grad_mask = grad_offset_all[:, grad_offset.shape[1]:, :, :]
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None, None, None, None)


class ModulatedDeformConv(nn.Module):
    r"""Applies an NPU based Modulated Deformable 2D convolution operation.

    The implementation of this ModulatedDeformConv is mainly based
    on the implementation of mmcv for design and reconstruction.

    In ModulatedDeformConvFunction, the forward and backward are customized,
    and the input tensor is reconstructed ito match the NPU based function.

    It is worth mentioning that DeformConv(DCNv1) is also implemented
    by setting modulated==False. Due to the difference between input
    and initialization, there is no additional implementation here.

    .. note::
        ModulatedDeformConv only implements operations under fp32 data types.
        Notice, weight and bias in conv_offset must be initialized to 0.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size(int, tuple): Size of the convolving kernel.
        stride(int, tuple): Stride of the convolution. Default: 1.
        padding (int or tuple): Zero-padding added to both sides of the input.
            Default: 0.
        dilation (int or tuple): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input.
            channels to output channels. Default: 1.
        deform_groups (int): Number of deformable group partitions.
        bias (bool): If True, adds a learnable bias to the output. Default: False.
        pack (bool): If True, conv_offset and mask will be included in this module. Default: True.

    Examples::
        >>> m = ModulatedDeformConv(32, 32, 1)
        >>> input_tensor = torch.randn(2, 32, 5, 5)
        >>> output = m(input_tensor)


        >>> x = torch.randn(2, 32, 7, 7)
        >>> model = ModulatedDeformConv(32, 32, 3, 2, 1)

        >>> torch.npu.set_device(0)
        >>> x = x.npu()
        >>> model = model.npu()

        >>> o = model(x)
        >>> l = o.sum()
        >>> l.backward()
        >>> print(l)
    """
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True,
                 pack=True,
                 ):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.pack = pack
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(self.weight.shape[0])
        if self.pack:
            self.conv_offset = nn.Conv2d(
                self.in_channels,
                self.deformable_groups * 3 * self.kernel_size[0] *
                self.kernel_size[1],
                kernel_size=self.kernel_size,
                stride=_pair(self.stride),
                padding=_pair(self.padding),
                bias=True)
        self.split_num = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        sort_index_for_npu = list(range(self.split_num))
        sort_index_for_npu_fp = sort_index_for_npu[1::2] + sort_index_for_npu[::2]
        sort_index_for_npu_bp_dict = {i: idx for idx, i in enumerate(sort_index_for_npu_fp)}
        sort_index_for_npu_bp = [sort_index_for_npu_bp_dict[i] for i in sort_index_for_npu]
        self.sort_index_for_npu_fp = torch.IntTensor(sort_index_for_npu_fp)
        self.sort_index_for_npu_bp = torch.IntTensor(sort_index_for_npu_bp)
        self.sort_index_for_npu_todevice = False
        self.init_param()

    def init_param(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        if self.pack:
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x):
        if self.pack:
            out = self.conv_offset(x)
            offset = out[:, :self.split_num, ...]
            mask = torch.sigmoid(out[:, self.split_num:, ...])
        else:
            x, offset, mask = x

        if not self.sort_index_for_npu_todevice:
            self.sort_index_for_npu_fp = self.sort_index_for_npu_fp.to(x.device)
            self.sort_index_for_npu_bp = self.sort_index_for_npu_bp.to(x.device)
            self.bias = self.bias.to(x.device)
            self.sort_index_for_npu_todevice = True

        return ModulatedDeformConv2dFunction.apply(
            x, offset, mask, self.weight, self.bias, self.with_bias,
            self.stride, self.padding, self.dilation,
            self.groups, self.deformable_groups,
            self.sort_index_for_npu_fp,
            self.sort_index_for_npu_bp,
        )


DCNv2 = ModulatedDeformConv