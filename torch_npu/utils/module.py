from typing import Optional
from statistics import mode
import warnings
import logging
import sys

import torch
import torch.nn.functional as F
import torch.distributed as pytorch_dist
from torch import Tensor
from torch.distributed.algorithms.join import Join
from torch.distributed import Reducer
from torch.nn.parameter import Parameter, UninitializedParameter, UninitializedBuffer
from torch.nn.modules.batchnorm import _NormBase, _LazyNormBase
from torch.nn.modules.module import Module
from torch.nn.parallel._functions import _streams

import torch_npu
from torch_npu.utils.syncbatchnorm import SyncBatchNorm as sync_batch_norm
import torch_npu.distributed as dist


def npu(self, device=None):
    r"""Moves all model parameters and buffers to the npu.

    This also makes associated parameters and buffers different objects. So
    it should be called before constructing optimizer if the module will
    live on npu while being optimized.

    Arguments:
        device (int, optional): if specified, all parameters will be
            copied to that device

    Returns:
        Module: self
    """
    device = torch.device("npu")
    if torch_npu.npu.is_available():
        with torch.no_grad():
            self.cast_weight(device)
    return self._apply(lambda t: t.npu(device))

def to(self, *args, **kwargs):
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

    if dtype is not None:
        if not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError('nn.Module.to only accepts floating point or complex '
                            'dtypes, but got desired dtype={}'.format(dtype))
        if dtype.is_complex:
            warnings.warn(
                "Complex modules are a new feature under active development whose design may change, "
                "and some modules might not work as expected when using complex tensors as parameters or buffers. "
                "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.md "
                "if a complex module does not work as expected.")
    if torch_npu.npu.is_available():
        with torch.no_grad():
            self.cast_weight(device)

    def convert(t):
        if convert_to_format is not None and t.dim() == 4:
            return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking, memory_format=convert_to_format)
        return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)

    return self._apply(convert)


def cast_weight(self, device):
    def _format_cast(module, class_name):
        if issubclass(class_name, torch.nn.Linear) and not torch.npu.get_mm_bmm_format_nd():
            module.weight.data = module.weight.data.to(device)
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29) # ACL_FORMAT_FRACTAL_NZ
        if "MultiheadAttention" in str(class_name) and \
                hasattr(module,"q_proj_weight") and module.q_proj_weight is not None and \
                hasattr(module,"k_proj_weight") and module.k_proj_weight is not None and \
                hasattr(module,"v_proj_weight") and module.v_proj_weight is not None and \
                not torch.npu.get_mm_bmm_format_nd():
            module.q_proj_weight.data = module.q_proj_weight.data.to(device)
            module.q_proj_weight.data = torch_npu.npu_format_cast(module.q_proj_weight.data, 29)
            module.k_proj_weight.data = module.k_proj_weight.data.to(device)
            module.k_proj_weight.data = torch_npu.npu_format_cast(module.k_proj_weight.data, 29)
            module.v_proj_weight.data = module.v_proj_weight.data.to(device)
            module.v_proj_weight.data = torch_npu.npu_format_cast(module.v_proj_weight.data, 29)

        if torch.npu.is_jit_compile_false():
            return
        if issubclass(class_name, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            if module.affine:
                module.weight.data = module.weight.data.to(device)
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 3)  # ACL_FORMAT_NC1HWC0
                module.bias.data = module.bias.data.to(device)
                module.bias.data = torch_npu.npu_format_cast(module.bias.data, 3)
            if module.track_running_stats:
                module.running_mean.data = module.running_mean.data.to(device)
                module.running_mean.data = torch_npu.npu_format_cast(module.running_mean.data, 3)
                module.running_var.data = module.running_var.data.to(device)
                module.running_var.data = torch_npu.npu_format_cast(module.running_var.data, 3)
        if issubclass(class_name, torch.nn.BatchNorm3d):
            # at present can not cast 1d to NDC1HWC0
            return
        if issubclass(class_name, torch.nn.Conv2d):
            if module.groups > 1:
                return
            if hasattr(module, "weight") and module.weight is not None and \
                "weight" in dict(module.named_parameters()):
                module.weight.data = module.weight.data.to(device)
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 4)  # ACL_FORMAT_FRACTAL_Z
        if issubclass(class_name, torch.nn.Conv3d):
            module.weight.data = module.weight.data.to(device)
            module.weight.data = torch_npu.npu_format_cast(module.weight.data.half(), 33).float()  # ACL_FRACTAL_Z_3D

    if device is None or not "npu" in str(device):
        return

    current_class = self.__class__
    _format_cast(self, current_class)

    if not self.children:
        return 

    for sub_module in self.children():
        if isinstance(sub_module, torch.nn.Module):
            sub_module.cast_weight(device)


def lstm_forward(self, input, hx=None):
    orig_input = input
    # xxx: isinstance check needs to be in conditional for TorchScript to compile
    if isinstance(orig_input, torch.nn.utils.rnn.PackedSequence):
        input, batch_sizes, sorted_indices, unsorted_indices = input
        max_batch_size = batch_sizes[0]
        max_batch_size = int(max_batch_size)
    else:
        batch_sizes = None
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None

    if hx is None:
        num_directions = 2 if self.bidirectional else 1
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        h_zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, real_hidden_size,
                                dtype=input.dtype, device=input.device)
        c_zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
        hx = (h_zeros, c_zeros)
    else:
        # Each batch of the hidden state should match the input sequence that
        # the user believes he/she is passing in.
        hx = self.permute_hidden(hx, sorted_indices)

    self.check_forward_args(input, hx, batch_sizes)
    if batch_sizes is None:
        result = torch._VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
                                self.dropout, self.training, self.bidirectional, self.batch_first)
    else:
        if batch_sizes.device != input.device:
            batch_sizes_npu = batch_sizes.to(input.device)
            result = torch._VF.lstm(input, batch_sizes_npu, hx, self._flat_weights, self.bias,
                                    self.num_layers, self.dropout, self.training, self.bidirectional)
            # 根据TMG决策，pack-lstm-pad时，保持有效T0时序内pad进行lstm定长计算，输出为pack且shape转换[T0*B, *]
            if isinstance(orig_input, torch.nn.utils.rnn.PackedSequence):
                result = list(result)
                shape = [result[0].shape[0] * result[0].shape[1]]
                if result[0].dim() > 2:
                    shape = shape + list(result[0].shape[2:])
                result[0] = result[0].reshape(shape)
                result = tuple(result)
        else:
            result = torch._VF.lstm(input, batch_sizes, hx, self._flat_weights, self.bias,
                                    self.num_layers, self.dropout, self.training, self.bidirectional)
    output = result[0]
    hidden = result[1:]
    # xxx: isinstance check needs to be in conditional for TorchScript to compile
    if isinstance(orig_input, torch.nn.utils.rnn.PackedSequence):
        output_packed = torch.nn.utils.rnn.PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output_packed, self.permute_hidden(hidden, unsorted_indices)
    else:
        return output, self.permute_hidden(hidden, unsorted_indices)


def syncbn_forward(self, input1: torch.Tensor) -> torch.Tensor:
    # currently only NPU or GPU input is supported
    if (not input1.is_cuda) and (not input1.is_npu):
        raise ValueError('SyncBatchNorm expected input tensor to be on NPU or GPU')

    self._check_input_dim(input1)
    self._check_non_zero_input_channels(input1)

    # exponential_average_factor is set to self.momentum
    # (when it is available) only so that it gets updated
    # in ONNX graph when this node is exported to ONNX.
    if self.momentum is None:
        exponential_average_factor = 0.0
    else:
        exponential_average_factor = self.momentum

    if self.training and self.track_running_stats:
        assert self.num_batches_tracked is not None
        self.num_batches_tracked.add_(1)
        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / self.num_batches_tracked.item()
        else:  # use exponential moving average
            exponential_average_factor = self.momentum

    r"""
    Decide whether the mini-batch stats should be used for normalization rather than the buffers.
    Mini-batch stats are used in training mode, and in eval mode when buffers are None.
    """
    if self.training:
        bn_training = True
    else:
        bn_training = (self.running_mean is None) and (self.running_var is None)

    r"""
    Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
    passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
    used for normalization (i.e. in eval mode when buffers are not None).
    """
    # If buffers are not to be tracked, ensure that they won't be updated
    running_mean = (
        self.running_mean if not self.training or self.track_running_stats else None
    )
    running_var = (
        self.running_var if not self.training or self.track_running_stats else None
    )

    # Don't sync batchnorm stats in inference mode (model.eval()).
    need_sync = (bn_training and self.training)
    if need_sync:
        process_group = torch.distributed.group.WORLD
        if self.process_group:
            process_group = self.process_group
        world_size = torch.distributed.get_world_size(process_group)
        need_sync = world_size > 1

    # fallback to framework BN when synchronization is not necessary
    if not need_sync:
        return F.batch_norm(
            input1, running_mean, running_var, self.weight, self.bias,
            bn_training, exponential_average_factor, self.eps)
    else:
        assert bn_training
        return sync_batch_norm.apply(
            input1, self.weight, self.bias, running_mean, running_var,
            self.eps, exponential_average_factor, process_group, world_size)


def _ddp_init_helper(
    self,
    parameters,
    expect_sparse_gradient,
    param_to_name_mapping,
    static_graph,):
    """
    Initialization helper function that does the following:
    (1) bucketing the parameters for reductions
    (2) resetting the bucketing states
    (3) registering the grad hooks
    (4) Logging construction-time DDP logging data
    (5) passing a handle of DDP to SyncBatchNorm Layer
    """
    if static_graph is True or self.find_unused_parameters is False:
        bucket_size_limits = [sys.maxsize]
    else:
        bucket_size_limits = [
            pytorch_dist._DEFAULT_FIRST_BUCKET_BYTES,
            self.bucket_bytes_cap,
        ]
    (bucket_indices, per_bucket_size_limits) = dist._compute_bucket_assignment_by_size(
        parameters,
        bucket_size_limits,
        expect_sparse_gradient)

    # Note: reverse list of buckets because we want to approximate the
    # order in which their gradients are produced, and assume they
    # are used in the forward pass in the order they are defined.
    self.reducer = dist.Reducer(
        parameters,
        list(reversed(bucket_indices)),
        list(reversed(per_bucket_size_limits)),
        self.process_group,
        expect_sparse_gradient,
        # The bucket size limit is specified in the constructor.
        # Additionally, we allow for a single small bucket for parameters
        # that are defined first, such that their gradients don't spill into
        # a much larger bucket, adding unnecessary latency after gradient
        # computation finishes. Experiments showed 1MB is a reasonable value.
        self.bucket_bytes_cap,
        self.find_unused_parameters,
        self.gradient_as_bucket_view,
        param_to_name_mapping,
        # User can set dist._DEFAULT_FIRST_BUCKET_BYTES to tune DDP first
        # bucket.
        pytorch_dist._DEFAULT_FIRST_BUCKET_BYTES)

    ori_reducer = Reducer(
        parameters,
        list(reversed(bucket_indices)),
        list(reversed(per_bucket_size_limits)),
        self.process_group,
        expect_sparse_gradient,
        # The bucket size limit is specified in the constructor.
        # Additionally, we allow for a single small bucket for parameters
        # that are defined first, such that their gradients don't spill into
        # a much larger bucket, adding unnecessary latency after gradient
        # computation finishes. Experiments showed 1MB is a reasonable value.
        self.bucket_bytes_cap,
        self.find_unused_parameters,
        self.gradient_as_bucket_view,
        param_to_name_mapping,
        # User can set dist._DEFAULT_FIRST_BUCKET_BYTES to tune DDP first
        # bucket.
        pytorch_dist._DEFAULT_FIRST_BUCKET_BYTES,
    )

    self.logger = pytorch_dist.Logger(ori_reducer)
    # Set as a weak reference to avoid reference cycle between
    # logger and reducer.
    self.reducer.set_logger(self.logger)

    has_sync_bn = False
    for submodule in self.module.modules():
        if isinstance(submodule, torch.nn.SyncBatchNorm):
            has_sync_bn = True
            break

    # Set logging data that can be got during construction time.
    self.logger.set_construction_data_and_log(
        self.module.__class__.__name__,
        [] if self.device_ids is None else self.device_ids,
        -1 if self.output_device is None else self.output_device,
        self.broadcast_buffers,
        has_sync_bn,
        static_graph,
    )

    # passing a handle to torch.nn.SyncBatchNorm layer
    self._passing_sync_batchnorm_handle(self.module)


def apply_module_patch():
    torch.nn.Module.npu = npu
    torch.nn.Module.to = to
    torch.nn.Module.cast_weight = cast_weight
    torch.nn.modules.rnn.LSTM.forward = lstm_forward
    torch.nn.modules.batchnorm.SyncBatchNorm.forward = syncbn_forward
