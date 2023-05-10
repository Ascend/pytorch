# Copyright (c) 2020 Huawei Technologies Co., Ltd
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


from typing import Optional
from statistics import mode
import warnings
import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.algorithms.join import Join
from torch.nn.parameter import Parameter, UninitializedParameter, UninitializedBuffer
from torch.nn.modules.batchnorm import _NormBase, _LazyNormBase
from torch.nn.modules.module import Module
from torch.nn.parallel._functions import _streams

import torch_npu
import torch_npu.distributed as dist
from torch_npu.utils.syncbatchnorm import SyncBatchNorm as sync_batch_norm


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
        # Ref [cast weight in single op mode]
        is_graph_mode = torch_npu.npu.is_graph_mode()
        if is_graph_mode:
            torch_npu.npu.disable_graph_mode()
        with torch.no_grad():
            self.cast_weight(device)
        if is_graph_mode:
            torch_npu.npu.enable_graph_mode()
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
            # Ref [cast weight in single op mode]
            is_graph_mode = torch_npu.npu.is_graph_mode()
            if is_graph_mode:
                torch_npu.npu.disable_graph_mode()
            with torch.no_grad():
                self.cast_weight(device)
            if is_graph_mode:
                torch_npu.npu.enable_graph_mode()

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


def layernorm_forward(self, input: torch.Tensor) -> torch.Tensor:
    if self.training or (not input.is_npu):
        return torch.nn.functional.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)
    else:
        return torch_npu.npu_layer_norm_eval(input, self.normalized_shape, self.weight, self.bias, self.eps)


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


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    max_seq_length = sequence.batch_sizes.size(0)
    if total_length is not None:
        if total_length < max_seq_length:
            raise ValueError("Expected total_length to be at least the length "
                             "of the longest sequence in input, but got "
                             "total_length={} and max sequence length being {}"
                             .format(total_length, max_seq_length))
        max_seq_length = total_length
    if sequence.batch_sizes.device != sequence.data.device:
        batch_sizes_npu = sequence.batch_sizes.to(sequence.data.device)
        padded_output, lengths = torch._VF._pad_packed_sequence(
            sequence.data, batch_sizes_npu, batch_first, padding_value, max_seq_length)
    else:
        padded_output, lengths = torch._VF._pad_packed_sequence(
            sequence.data, sequence.batch_sizes, batch_first, padding_value, max_seq_length)
    unsorted_indices = sequence.unsorted_indices
    if unsorted_indices is not None:
        batch_dim = 0 if batch_first else 1
        return padded_output.index_select(batch_dim, unsorted_indices), lengths[unsorted_indices]
    return padded_output, lengths


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


def _normbase_init_(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True,
                    track_running_stats: bool = True, device=None, dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super(_NormBase, self).__init__()
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    self.track_running_stats = track_running_stats
    if self.affine:
        self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
        self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
    else:
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)
    if self.track_running_stats:
        self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
        self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
        self.running_mean: Optional[Tensor]
        self.running_var: Optional[Tensor]
        self.register_buffer('num_batches_tracked',
                                torch.tensor(0, dtype=torch.int32,
                                            **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        self.num_batches_tracked: Optional[Tensor]
    else:
        self.register_buffer('running_mean', None)
        self.register_buffer('running_var', None)
        self.register_buffer('num_batches_tracked', None)
    self.reset_parameters()


def _normbase__load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                    missing_keys, unexpected_keys, error_msgs):
    version = local_metadata.get("version", None)

    if (version is None or version < 2) and self.track_running_stats:
        # at version 2: added num_batches_tracked buffer
        #               this should have a default value of 0
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key not in state_dict:
            state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.int32)

    super(_NormBase, self)._load_from_state_dict(
        state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs)


def _lazynormbase__init__(self, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                          device=None, dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super(_LazyNormBase, self).__init__(
        # affine and track_running_stats are hardcoded to False to
        # avoid creating tensors that will soon be overwritten.
        0, eps, momentum, False, False, **factory_kwargs)
    self.affine = affine
    self.track_running_stats = track_running_stats
    if self.affine:
        self.weight = UninitializedParameter(**factory_kwargs)
        self.bias = UninitializedParameter(**factory_kwargs)
    if self.track_running_stats:
        self.running_mean = UninitializedBuffer(**factory_kwargs)
        self.running_var = UninitializedBuffer(**factory_kwargs)
        self.num_batches_tracked = torch.tensor(
            0, dtype=torch.int32, **{k: v for k, v in factory_kwargs.items() if k != 'dtype'})


def apply_module_patch():
    torch.nn.Module.npu = npu
    torch.nn.Module.to = to
    torch.nn.Module.cast_weight = cast_weight
    torch.nn.LayerNorm.forward = layernorm_forward
    torch.nn.modules.rnn.LSTM.forward = lstm_forward
    torch.nn.utils.rnn.pad_packed_sequence = pad_packed_sequence
    torch.nn.modules.batchnorm.SyncBatchNorm.forward = syncbn_forward
    torch.nn.modules.batchnorm._NormBase.__init__ = _normbase_init_
    torch.nn.modules.batchnorm._NormBase._load_from_state_dict = _normbase__load_from_state_dict
    torch.nn.modules.batchnorm._LazyNormBase.__init__ = _lazynormbase__init__
