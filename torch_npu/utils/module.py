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


from statistics import mode
import warnings
import logging
import torch
import torch.nn.functional as F

from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
from torch_npu.utils.tensor_methods import torch_device_guard

import torch_npu


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

@torch_device_guard
def to(self, *args, **kwargs):
    if args and isinstance(args[0], str) and 'npu' in args[0]:
        args = tuple([list(args)[0].replace('npu', torch_npu.npu.native_device)])
    if kwargs and 'npu' in kwargs.get("device", ""):
        kwargs['device'] = kwargs['device'].replace("npu", torch_npu.npu.native_device)
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
        if issubclass(class_name, (torch.nn.BatchNorm3d, torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
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
        if "MultiheadAttention" in str(class_name) and \
            hasattr(module,"q_proj_weight") and module.q_proj_weight is not None and \
            hasattr(module,"k_proj_weight") and module.k_proj_weight is not None and \
            hasattr(module,"v_proj_weight") and module.v_proj_weight is not None:
            module.q_proj_weight.data = module.q_proj_weight.data.to(device)
            module.q_proj_weight.data = torch_npu.npu_format_cast(module.q_proj_weight.data, 29)
            module.k_proj_weight.data = module.k_proj_weight.data.to(device)
            module.k_proj_weight.data = torch_npu.npu_format_cast(module.k_proj_weight.data, 29)
            module.v_proj_weight.data = module.v_proj_weight.data.to(device)
            module.v_proj_weight.data = torch_npu.npu_format_cast(module.v_proj_weight.data, 29)

    # supported devices list: "npu"(from module.npu), "xla"(from module.to)
    support_cast_devices = [torch_npu.npu.native_device, torch_npu.npu.npu_device]
    if device is None or not any(support_cast_device in str(device) for support_cast_device in support_cast_devices):
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


def ddp_forward(self, *inputs, **kwargs):
    if self.ddp_uneven_inputs_config.ddp_join_enabled:
        ones = torch.ones(
            1, device=self.device
        )
        work = torch_npu.distributed.all_reduce(ones, group=self.process_group, async_op=True)
        self.reducer._set_forward_pass_work_handle(
            work, self.ddp_uneven_inputs_config.ddp_join_divide_by_initial_world_size
        )

    # Calling _rebuild_buckets before forward compuation,
    # It may allocate new buckets before deallocating old buckets
    # inside _rebuild_buckets. To save peak memory usage,
    # call _rebuild_buckets before the peak memory usage increases
    # during forward computation.
    # This should be called only once during whole training period.
    if self.reducer._rebuild_buckets():
        logging.info("Reducer buckets have been rebuilt in this iteration.")

    if self.require_forward_param_sync:
        self._sync_params()

    if self.ddp_uneven_inputs_config.ddp_join_enabled:
        # Notify joined ranks whether they should sync in backwards pass or not.
        self._check_global_requires_backward_grad_sync(is_joined_rank=False)
    # Note: module.device_type was builded from device.type("npu") inside Class Module
    if self.device_ids and self.device_type != torch_npu.npu.npu_device:
        if len(self.device_ids) == 1:
            inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
            output = self.module(*inputs[0], **kwargs[0])
        else:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
            output = self.gather(outputs, self.output_device)
    else:
        output = self.module(*inputs, **kwargs)

    if torch.is_grad_enabled() and self.require_backward_grad_sync:
        self.require_forward_param_sync = True
        # We'll return the output object verbatim since it is a freeform
        # object. We need to find any tensors in this object, though,
        # because we need to figure out which parameters were used during
        # this forward pass, to ensure we short circuit reduction for any
        # unused parameters. Only if `find_unused_parameters` is set.
        if self.find_unused_parameters:
            self.reducer.prepare_for_backward(list(torch.nn.parallel.distributed._find_tensors(output)))
        else:
            self.reducer.prepare_for_backward([])
    else:
        self.require_forward_param_sync = False

    return output


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

    # exponential_average_factor is set to self.momentum
    # (when it is available) only so that it gets updated
    # in ONNX graph when this node is exported to ONNX.
    if self.momentum is None:
        exponential_average_factor = 0.0
    else:
        exponential_average_factor = self.momentum

    if self.training and self.track_running_stats:
        assert self.num_batches_tracked is not None
        self.num_batches_tracked = self.num_batches_tracked + 1
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
    assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
    assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
    running_mean = self.running_mean if not self.training or self.track_running_stats else None
    running_var = self.running_var if not self.training or self.track_running_stats else None

    need_sync = bn_training
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
        if not self.ddp_gpu_size:
            raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')

        assert bn_training
        return sync_batch_norm.apply(
            input1, self.weight, self.bias, running_mean, running_var,
            self.eps, exponential_average_factor, process_group, world_size)

def apply_module_patch():
    torch.nn.Module.npu = npu
    torch.nn.Module.to = to
    torch.nn.Module.cast_weight = cast_weight
    torch.nn.LayerNorm.forward = layernorm_forward
    torch.nn.parallel.DistributedDataParallel.forward = ddp_forward
    torch.nn.modules.rnn.LSTM.forward = lstm_forward
    torch.nn.utils.rnn.pad_packed_sequence = pad_packed_sequence
    torch.nn.modules.batchnorm.SyncBatchNorm.forward = syncbn_forward
