import numpy as np
import torch
import torch.nn as nn
import torch_npu


class ChannelShuffle(nn.Module):
    r"""Applies an NPU compatible channel shuffle operation.

    In order to avoid contiguous operation which is not efficient on npu, we replaced the original operation
    with a rewrite of the same semantics. Two discontinuous operations are replaced, transpose and chunk.

    .. note::
        Only group=2 is implemented, modify other group scenarios yourself.

    Args:
        in_channels (int): The total number of channels in the input tensors
        groups (int): The number of shuffle groups. Default: 2
        split_shuffle (bool): Whether to execute the chunk after shuffle. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`, `(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})`

    Examples::
        >>> x1 = torch.randn(2,32,7,7)
        >>> x2 = torch.randn(2,32,7,7)
        >>> m = ChannelShuffle(64, split_shuffle=True)
        >>> output = m(x1, x2)

    """

    def __init__(self, in_channels, groups=2, split_shuffle=True):
        super(ChannelShuffle, self).__init__()
        self.split_shuffle = split_shuffle
        self.group_len = in_channels // groups

        # init out_channels
        self.out_channels = np.array(list(range(in_channels))).reshape(groups, self.group_len).transpose(1, 0).flatten()
        self.out_channels = torch.from_numpy(self.out_channels).long()

        # init index used in fp & bp
        # Only group=2 is implemented, modify other group scenarios yourself.
        if self.split_shuffle:
            self.fp_index1 = self.out_channels[:self.group_len]
            self.fp_index2 = self.out_channels[self.group_len:]
        else:
            self.fp_index = self.out_channels
        self.bp_index1 = torch.tensor(list(range(0, in_channels, 2)))
        self.bp_index2 = torch.tensor(list(range(1, in_channels, 2)))

        self.checked = False

    def check_self(self, x):
        r"""Check device equipment between tensors.
        """
        if self.bp_index1.device == x.device:
            self.checked = True
            return

        device = x.device

        if str(device).startswith('npu'):
            if self.split_shuffle:
                self.fp_index1 = self.fp_index1.int()
                self.fp_index2 = self.fp_index2.int()
            else:
                self.fp_index = self.fp_index.int()
            self.bp_index1 = self.bp_index1.int()
            self.bp_index2 = self.bp_index2.int()

        if self.split_shuffle:
            self.fp_index1 = self.fp_index1.to(device)
            self.fp_index2 = self.fp_index2.to(device)
        else:
            self.fp_index = self.fp_index.to(device)
        self.bp_index1 = self.bp_index1.to(device)
        self.bp_index2 = self.bp_index2.to(device)

    def forward(self, x1, x2):
        if not self.checked:
            self.check_self(x1)
        if self.split_shuffle:
            if self.training:
                output = IndexSelectHalfImplementation.apply(x1, x2, self.fp_index1, self.fp_index2, self.bp_index1,
                                                           self.bp_index2)
            else:
                output = indexselect_half_implementation_forward(x1, x2, self.fp_index1, self.fp_index2)
        else:
            if self.training:
                output = IndexSelectFullImplementation.apply(x1, x2, self.fp_index, self.bp_index1, self.bp_index2)
            else:
                output = indexselect_full_implementation_forward(x1, x2, self.fp_index)
        return output


def indexselect_full_implementation_forward(x1, x2, fp_index):
    x = torch.cat([x1, x2], dim=1)
    result = x.index_select(1, fp_index)
    return result


def indexselect_half_implementation_forward(x1, x2, fp_index1, fp_index2):
    x = torch.cat([x1, x2], dim=1)
    return x.index_select(1, fp_index1), x.index_select(1, fp_index2)


class IndexSelectFullImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, fp_index, bp_index1, bp_index2):
        if str(x1.device).startswith('npu'):
            # for training stream stable
            stream = torch.npu.current_stream()
            stream.synchronize()

        ctx.bp_index1 = bp_index1
        ctx.bp_index2 = bp_index2
        x = torch.cat([x1, x2], dim=1)
        result = x.index_select(1, fp_index)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if str(grad_output.device).startswith('npu'):
            # for training stream stable
            stream = torch.npu.current_stream()
            stream.synchronize()
            # convert to NCHW to avoid extra 5HD --> 4D
            grad_output.data = torch_npu.npu_format_cast(grad_output.data, 0)

        out1 = grad_output.index_select(1, ctx.bp_index1)
        out2 = grad_output.index_select(1, ctx.bp_index2)
        return out1, out2, None, None, None, None


class IndexSelectHalfImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, fp_index1, fp_index2, bp_index1, bp_index2):
        ctx.bp_index1 = bp_index1
        ctx.bp_index2 = bp_index2
        x = torch.cat([x1, x2], dim=1)
        return x.index_select(1, fp_index1), x.index_select(1, fp_index2)

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        grad_output = torch.cat([grad_output1, grad_output2], 1)
        out1 = grad_output.index_select(1, ctx.bp_index1)
        out2 = grad_output.index_select(1, ctx.bp_index2)
        return out1, out2, None, None, None, None