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
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
from torch.nn.utils.rnn import PackedSequence
from torch._C._nn import _parse_to as torch_parse_to

import torch_npu
import torch_npu.distributed as dist
from torch_npu.utils.syncbatchnorm import SyncBatchNorm as sync_batch_norm
from torch_npu.utils.tensor_methods import torch_device_guard

logger = logging.getLogger(__name__)
origin_mpdl_iter_init = _MultiProcessingDataLoaderIter.__init__


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
                "and some modules might not work as expected when using complex tensors as parameters or buffers. ")
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
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)  # ACL_FORMAT_FRACTAL_NZ

        if issubclass(class_name, torch.nn.MultiheadAttention) and \
                module.q_proj_weight is not None and \
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
        if issubclass(class_name, torch.nn.LazyConv3d):
            return
        if issubclass(class_name, torch.nn.Conv3d):
            module.weight.data = module.weight.data.to(device)
            module.weight.data = torch_npu.npu_format_cast(module.weight.data.half(), 33).float()  # ACL_FRACTAL_Z_3D

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


def layernorm_forward(self, input1: torch.Tensor) -> torch.Tensor:
    if self.training or (not input1.is_npu):
        return torch.nn.functional.layer_norm(
            input1, self.normalized_shape, self.weight, self.bias, self.eps)
    else:
        return torch_npu.npu_layer_norm_eval(input1, self.normalized_shape, self.weight, self.bias, self.eps)


def ddp_forward(self, *inputs, **kwargs):
    with torch.autograd.profiler.record_function("DistributedDataParallel.forward"):
        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.num_iterations += 1
            self.reducer.prepare_for_forward()

        # Notify the join context that this process has not joined, if
        # needed
        work = Join.notify_join_context(self)
        if work:
            self.reducer._set_forward_pass_work_handle(work, self._divide_by_initial_world_size)

        # Calling _rebuild_buckets before forward compuation,
        # It may allocate new buckets before deallocating old buckets
        # inside _rebuild_buckets. To save peak memory usage,
        # call _rebuild_buckets before the peak memory usage increases
        # during forward computation.
        # This should be called only once during whole training period.
        if torch.is_grad_enabled() and self.reducer._rebuild_buckets():
            logger.info("Reducer buckets have been rebuilt in this iteration.")
            self._has_rebuilt_buckets = True

        # sync params according to location (before/after forward) user
        # specified as part of hook, if hook was specified.
        buffer_hook_registered = hasattr(self, 'buffer_hook')
        if self._check_sync_bufs_pre_fwd():
            self._sync_buffers()

        if self._join_config.enable:
            # Notify joined ranks whether they should sync in backwards pass or not.
            self._check_global_requires_backward_grad_sync(is_joined_rank=False)

        output = self.module(*inputs, **kwargs)

        # sync params according to location (before/after forward) user
        # specified as part of hook, if hook was specified.
        if self._check_sync_bufs_post_fwd():
            self._sync_buffers()

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if self.find_unused_parameters and not self.static_graph:
                # Do not need to populate this for static graph.:
                self.reducer.prepare_for_backward(list(torch.nn.parallel.distributed._find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False

    return output


def mpdl_iter_init(self, *args, **kwargs):
    torch_npu.npu.synchronize()
    origin_mpdl_iter_init(self, *args, **kwargs)


def lstm_forward(self, input1, hx=None):
    orig_input = input1
    # isinstance check needs to be in conditional for TorchScript to compile
    batch_sizes = None
    if isinstance(orig_input, torch.nn.utils.rnn.PackedSequence):
        input1, batch_sizes, sorted_indices, unsorted_indices = input1
        max_batch_size = batch_sizes[0]
        max_batch_size = int(max_batch_size)
    else:
        batch_sizes = None
        is_batched = input1.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        if not is_batched:
            input1 = input1.unsqueeze(batch_dim)
        max_batch_size = input1.size(0) if self.batch_first else input1.size(1)
        sorted_indices = None
        unsorted_indices = None

    if hx is None:
        num_directions = 2 if self.bidirectional else 1
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        h_zeros = torch.zeros(self.num_layers * num_directions,
                              max_batch_size, real_hidden_size,
                              dtype=input1.dtype, device=input1.device)
        c_zeros = torch.zeros(self.num_layers * num_directions,
                              max_batch_size, self.hidden_size,
                              dtype=input1.dtype, device=input1.device)
        hx = (h_zeros, c_zeros)
    else:
        if batch_sizes is None:
            if is_batched:
                if (hx[0].dim() != 3 or hx[1].dim() != 3):
                    msg = ("For batched 3-D input, hx and cx should "
                           f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
                    raise RuntimeError(msg)
            else:
                if hx[0].dim() != 2 or hx[1].dim() != 2:
                    msg = ("For unbatched 2-D input, hx and cx should "
                           f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
                    raise RuntimeError(msg)
                hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))

        # Each batch of the hidden state should match the input sequence that
        # the user believes he/she is passing in.
        hx = self.permute_hidden(hx, sorted_indices)

    self.check_forward_args(input1, hx, batch_sizes)
    if batch_sizes is None:
        result = torch._VF.lstm(input1, hx, self._flat_weights, self.bias, self.num_layers,
                                self.dropout, self.training, self.bidirectional, self.batch_first)
    else:
        if batch_sizes.device != input1.device:
            batch_sizes_npu = batch_sizes.to(input1.device)
            result_tmp = torch._VF.lstm(input1, batch_sizes_npu, hx, self._flat_weights, self.bias,
                                        self.num_layers, self.dropout, self.training, self.bidirectional)
            # when pack-lstm-pad，remain valid pads in T0 because lstm can only support fixed length in npu.
            if isinstance(orig_input, torch.nn.utils.rnn.PackedSequence):
                shape = [result_tmp[0].shape[0] * result_tmp[0].shape[1]]
                if result_tmp[0].dim() > 2:
                    shape = shape + list(result_tmp[0].shape[2:])
                result = (result_tmp[0].reshape(shape),) + result_tmp[1:]
        else:
            result = torch._VF.lstm(input1, batch_sizes, hx, self._flat_weights, self.bias,
                                    self.num_layers, self.dropout, self.training, self.bidirectional)
    output = result[0]
    hidden = result[1:]
    # isinstance check needs to be in conditional for TorchScript to compile
    if isinstance(orig_input, torch.nn.utils.rnn.PackedSequence):
        output_packed = torch.nn.utils.rnn.PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output_packed, self.permute_hidden(hidden, unsorted_indices)
    else:
        if not is_batched:
            output = output.squeeze(batch_dim)
            hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
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
        if self.num_batches_tracked is None:
            raise ValueError("self.num_batches_tracked is None")
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
        if not bn_training:
            raise ValueError("not bn_training")
        return sync_batch_norm.apply(
            input1, self.weight, self.bias, running_mean, running_var,
            self.eps, exponential_average_factor, process_group, world_size)


def DDPJoinHook__init__(self, ddp, divide_by_initial_world_size):
    """
    Sets config variables for internal usage.
    """
    if not isinstance(ddp, torch.nn.parallel.DistributedDataParallel):
        raise TypeError("DDP join hook requires passing in a DistributedDataParallel "
                        "instance as the state")
    self.ddp = ddp
    self.ddp._divide_by_initial_world_size = divide_by_initial_world_size
    super(torch.nn.parallel.distributed._DDPJoinHook, self).__init__()


def ddp_ddp_init_helper(
        self, parameters, expect_sparse_gradient, param_to_name_mapping):
    """
    Initialization helper function that does the following:
    (1) bucketing the parameters for reductions
    (2) resetting the bucketing states
    (3) registering the grad hooks
    (4) Logging constructin-time DDP logging data
    (5) passing a handle of DDP to SyncBatchNorm Layer
    """
    self.num_iterations = 0
    # The bucket size limit is specified in the constructor.
    # Additionally, we allow for a single small bucket for parameters
    # that are defined first, such that their gradients don't spill into
    # a much larger bucket, adding unnecessary latency after gradient
    # computation finishes. Experiments showed 1MB is a reasonable value.
    bucket_indices, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
        parameters,
        [dist._DEFAULT_FIRST_BUCKET_BYTES, self.bucket_bytes_cap],
        expect_sparse_gradient,
    )

    # Note: reverse list of buckets because we want to approximate the
    # order in which their gradients are produced, and assume they
    # are used in the forward pass in the order they are defined.
    self.reducer = dist.Reducer(
        parameters,
        list(reversed(bucket_indices)),
        list(reversed(per_bucket_size_limits)),
        self.process_group,
        expect_sparse_gradient,
        self.bucket_bytes_cap,
        self.find_unused_parameters,
        self.gradient_as_bucket_view,
        param_to_name_mapping,
        # User can set dist._DEFAULT_FIRST_BUCKET_BYTES to tune DDP first
        # bucket.
        dist._DEFAULT_FIRST_BUCKET_BYTES
    )

    # don't support logger
    self.logger = None

    has_sync_bn = False
    for submodule in self.module.modules():
        if isinstance(submodule, torch.nn.SyncBatchNorm):
            has_sync_bn = True
            break

    # passing a handle to torch.nn.SyncBatchNorm layer
    self._passing_sync_batchnorm_handle(self.module)


def ddp__setstate__(self, state):
    # If serializable, then the process group should be the default one
    self.process_group = torch_npu.distributed.distributed_c10d._get_default_group()
    Module.__setstate__(self, state)
    self.__dict__.setdefault("require_forward_param_sync", True)
    self.__dict__.setdefault("require_backward_grad_sync", True)
    parameters, expect_sparse_gradient = self._build_params_for_reducer()
    # In debug mode, build a mapping of parameter index -> parameter.
    param_to_name_mapping = {}
    # Builds reducer
    self._ddp_init_helper(parameters, expect_sparse_gradient, param_to_name_mapping)
    if self.static_graph:
        self.reducer._set_static_graph()


def ddp_register_builtin_comm_hook(self, comm_hook_type):
    r"""
    Registers a built-in communication hook that specifies how DDP
    aggregates gradients across multiple workers.
    The built-in hooks aim to provide efficient C++ implementations for certain hooks,
    which might not be as efficient if implemented in Python using a Python communication hook.

    Args:
        comm_hook_type (dist.BuiltinCommHookType): type of communication hook, such as ALLREDUCE, FP16_COMPRESS, etc.

    .. warning ::
        DDP communication hook can only be registered once and should be registered
        before calling backward.

    Example::
        Below is an example of a FP16 compression where gradients are
        compressed into 16-bit floating-point numbers before allreduce, and
        then decompressed after allreduce.

        >>> ddp._register_builtin_comm_hook(dist.BuiltinCommHookType.FP16_COMPRESS)

    """
    dist._register_builtin_comm_hook(self.reducer, comm_hook_type)


def ddp_get_ddp_logging_data(self):
    r"""
    This interface can be called after DistributedDataParallel() is
    constructed. It returns a dictionary of logging data. It could help
    for debugging and analysis. The loggind data includes DistributedDataParallel
    constructor input parameters, some internal states of DistributedDataParallel
    and performance metrics. Simply print the dictorinary and see what
    these metrics are.
    This is a prototype interface and subject to change in the future.
    """
    raise AttributeError('ddp is not supported get_ddp_logging_data')


def ddp_set_static_graph(self):
    """
    It is recommended to set static graph in the DDP constructor, which will
    call this private API internally.
    """
    # If self.static_graph has been set, no need to set it again
    if self.static_graph:
        warnings.warn(
            "You've set static_graph to be True, no need to set it again.")
        return
    self.static_graph = True
    self.reducer._set_static_graph()
    if self.find_unused_parameters:
        warnings.warn(
            "You passed find_unused_parameters=true to DistributedDataParallel, "
            "`_set_static_graph` will detect unused parameters automatically, so "
            "you do not need to set find_unused_parameters=true, just be sure these "
            "unused parameters will not change during training loop while calling "
            "`_set_static_graph`."
        )


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


def _get_stream(device: int):
    """Gets a background stream for copying between CPU and NPU"""
    global _streams
    if device == -1:
        return None
    if _streams is None:
        _streams = [None] * torch.npu.device_count()
    if _streams[device] is None:
        _streams[device] = torch.npu.Stream(device)
    return _streams[device]


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


def gru_forward(self, input_tensor, hx=None):
    orig_input = input_tensor

    if isinstance(orig_input, PackedSequence):
        input_tensor, batch_sizes, sorted_indices, unsorted_indices = input_tensor
        max_batch_size = batch_sizes[0]
        max_batch_size = int(max_batch_size)
    else:
        batch_sizes = None
        is_batched = input_tensor.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        if not is_batched:
            input_tensor = input_tensor.unsqueeze(batch_dim)
            if hx is not None:
                if hx.dim() != 2:
                    raise RuntimeError(
                        f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor")
                hx = hx.unsqueeze(1)
        else:
            if hx is not None and hx.dim() != 3:
                raise RuntimeError(
                    f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor")
        max_batch_size = input_tensor.size(0) if self.batch_first else input_tensor.size(1)
        sorted_indices = None
        unsorted_indices = None

    if hx is None:
        num_directions = 2 if self.bidirectional else 1
        hx = torch.zeros(self.num_layers * num_directions,
                         max_batch_size, self.hidden_size,
                         dtype=input_tensor.dtype, device=input_tensor.device)
    else:
        # Each batch of the hidden state should match the input sequence that
        # the user believes he/she is passing in.
        hx = self.permute_hidden(hx, sorted_indices)

    self.check_forward_args(input_tensor, hx, batch_sizes)
    if batch_sizes is None:
        result = torch._VF.gru(input_tensor, hx, self._flat_weights, self.bias, self.num_layers,
                               self.dropout, self.training, self.bidirectional, self.batch_first)
    else:
        if batch_sizes.device != input_tensor.device:
            # convert to compact length
            start = 0
            idx_list = []
            batch_list = batch_sizes.numpy()
            for i in batch_list:
                idx = list(range(start, start + i, 1))
                idx_list = idx_list + idx
                start = start + batch_list[0]
            input_pack = input_tensor
            if len(idx_list) != input_tensor.shape[0]:
                idx_tensor = torch.Tensor(idx_list).long().to(input_tensor.device)
                input_pack = torch.nn.functional.embedding(idx_tensor, input_tensor)

            result = torch._VF.gru(input_pack, batch_sizes, hx, self._flat_weights, self.bias,
                                   self.num_layers, self.dropout, self.training, self.bidirectional)

            # convert to fixed length
            if len(idx_list) != input_tensor.shape[0]:
                start = 0
                cur = start
                shape = [1] + list(result[0].shape[1:])
                pad_tensor = torch.zeros(shape, device=input_tensor.device)
                cat_list = []
                for i in batch_list:
                    if (i < batch_list[0]):
                        slice_tensor = result[0][start: cur + i, :]
                        start = cur + i
                        cur = start
                        cat_list.append(slice_tensor)
                        for j in range(batch_list[0] - i):
                            cat_list.append(pad_tensor)
                    else:
                        cur = cur + batch_list[0]
                result0 = torch.cat(cat_list, 0)
                result = (result0, result[1])
        else:
            result = torch._VF.gru(input_tensor, batch_sizes, hx, self._flat_weights, self.bias,
                                   self.num_layers, self.dropout, self.training, self.bidirectional)
    output = result[0]
    hidden = result[1]

    if isinstance(orig_input, PackedSequence):
        output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output_packed, self.permute_hidden(hidden, unsorted_indices)
    else:
        if not is_batched:
            output = output.squeeze(batch_dim)
            hidden = hidden.squeeze(1)
        return output, self.permute_hidden(hidden, unsorted_indices)


@torch_device_guard
def _parse_to(*args, **kwargs):
    return torch_parse_to(*args, **kwargs)


def apply_module_patch():
    torch.nn.Module.npu = npu
    torch.nn.Module.to = to
    torch.nn.Module.cast_weight = cast_weight
    torch.nn.LayerNorm.forward = layernorm_forward
    torch.nn.parallel.distributed._DDPJoinHook.__init__ = DDPJoinHook__init__
    torch.nn.parallel.DistributedDataParallel.__setstate__ = ddp__setstate__
    torch.nn.parallel.DistributedDataParallel._ddp_init_helper = ddp_ddp_init_helper
    torch.nn.parallel.DistributedDataParallel._get_ddp_logging_data = ddp_get_ddp_logging_data
    torch.nn.parallel.DistributedDataParallel._register_builtin_comm_hook = ddp_register_builtin_comm_hook
    torch.nn.parallel.DistributedDataParallel._set_static_graph = ddp_set_static_graph
    torch.nn.parallel.DistributedDataParallel.forward = ddp_forward
    torch.utils.data.dataloader._MultiProcessingDataLoaderIter.__init__ = mpdl_iter_init
    torch.nn.modules.rnn.LSTM.forward = lstm_forward
    torch.nn.modules.rnn.GRU.forward = gru_forward
    torch.nn.utils.rnn.pad_packed_sequence = pad_packed_sequence
    torch.nn.modules.batchnorm.SyncBatchNorm.forward = syncbn_forward
    torch.nn.modules.batchnorm._NormBase.__init__ = _normbase_init_
    torch.nn.modules.batchnorm._NormBase._load_from_state_dict = _normbase__load_from_state_dict
    torch.nn.modules.batchnorm._LazyNormBase.__init__ = _lazynormbase__init__
    torch.nn.parallel._functions._get_stream = _get_stream
    torch._C._nn._parse_to = _parse_to
