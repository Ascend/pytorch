from itertools import chain
from typing import Any, Dict, List, Optional, Sequence, Union, cast
from statistics import mode
import threading
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
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
from torch._utils import _get_device_index, _get_all_device_indices, _get_available_device_type, ExceptionWrapper
from torch.nn.parallel.parallel_apply import get_a_var
from torch.nn.parallel.scatter_gather import gather, scatter_kwargs
from torch.nn.parallel.replicate import replicate

import torch_npu
from torch_npu.npu.amp.autocast_mode import autocast
from torch_npu.npu.utils import get_device_name
from torch_npu.utils.syncbatchnorm import SyncBatchNorm as sync_batch_norm
from torch_npu.utils._error_code import ErrCode, pta_error

origin_mpdl_iter_init = _MultiProcessingDataLoaderIter.__init__

CONV3D_SUPPORT_FP32_SOC_PREFIX = ["Ascend910B", "Ascend910_93"]


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
                            'dtypes, but got desired dtype={}'.format(dtype) + pta_error(ErrCode.TYPE))
        if dtype.is_complex:
            warnings.warn(
                "Complex modules are a new feature under active development whose design may change, "
                "and some modules might not work as expected when using complex tensors as parameters or buffers. ")
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
            device_name = get_device_name()
            if any(device_name.startswith(prefix) for prefix in CONV3D_SUPPORT_FP32_SOC_PREFIX):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 33)
                return
            module.weight.data = torch_npu.npu_format_cast(module.weight.data.half(), 33).float()  # ACL_FRACTAL_Z_3D

    if device is None or "npu" not in str(device):
        return

    current_class = self.__class__
    _format_cast(self, current_class)

    if not self.children:
        return

    for sub_module in self.children():
        if isinstance(sub_module, torch.nn.Module):
            sub_module.cast_weight(device)


def _lstm_forward(self, input1, hx=None):
    orig_input = input1
    if isinstance(orig_input, torch.nn.utils.rnn.PackedSequence):
        input1, batch_sizes, sorted_indices, unsorted_indices = input1
        max_batch_size = batch_sizes[0]
        max_batch_size = int(max_batch_size)
    else:
        batch_sizes = None
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
            # pack-lstm-pad时，保持有效T0时序内pad进行lstm定长计算，输出为pack且shape转换[T0*B, *]
            if isinstance(orig_input, torch.nn.utils.rnn.PackedSequence):
                shape = [result_tmp[0].shape[0] * result_tmp[0].shape[1]]
                if result_tmp[0].dim() > 2:
                    shape = shape + list(result_tmp[0].shape[2:])
                result = (result_tmp[0].reshape(shape), ) + result_tmp[1:]
        else:
            result = torch._VF.lstm(input1, batch_sizes, hx, self._flat_weights, self.bias,
                                    self.num_layers, self.dropout, self.training, self.bidirectional)
    output = result[0]
    hidden = result[1:]

    if isinstance(orig_input, torch.nn.utils.rnn.PackedSequence):
        output_packed = torch.nn.utils.rnn.PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output_packed, self.permute_hidden(hidden, unsorted_indices)
    else:
        return output, self.permute_hidden(hidden, unsorted_indices)


def _syncbn_forward(self, input1: torch.Tensor) -> torch.Tensor:
    # currently only NPU or GPU input is supported
    if (not input1.is_cuda) and (not input1.is_npu):
        raise ValueError('SyncBatchNorm expected input tensor to be on NPU or GPU' + pta_error(ErrCode.VALUE))

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
            raise ValueError("self.num_batches_tracked is None" + pta_error(ErrCode.VALUE))
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
            raise ValueError("not bn_training" + pta_error(ErrCode.VALUE))
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
    (bucket_indices, per_bucket_size_limits) = torch_npu.distributed._compute_bucket_assignment_by_size(
        parameters,
        bucket_size_limits,
        expect_sparse_gradient)

    # Note: reverse list of buckets because we want to approximate the
    # order in which their gradients are produced, and assume they
    # are used in the forward pass in the order they are defined.
    self.reducer = torch_npu.distributed.Reducer(
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


def _mpdl_iter_init(self, *args, **kwargs):
    try:
        torch_npu.npu.synchronize()
    except:
        pass
    origin_mpdl_iter_init(self, *args, **kwargs)


def _parallel_apply(
    modules: Sequence[Module],
    inputs: Sequence[Any],
    kwargs_tup: Optional[Sequence[Dict[str, Any]]] = None,
    devices: Optional[Sequence[Optional[Union[int, torch.device]]]] = None,
) -> List[Any]:
    if len(modules) != len(inputs):
        raise AssertionError(
            f'The number of modules {len(modules)} is not equal to the number of inputs {len(inputs)}' +
            pta_error(ErrCode.PARAM))
    if kwargs_tup is not None:
        if len(modules) != len(kwargs_tup):
            raise AssertionError(
                f'The number of modules {len(modules)} is not equal to the number of kwargs_tup {len(kwargs_tup)}' +
                pta_error(ErrCode.PARAM))
    else:
        kwargs_tup = (cast(Dict[str, Any], {}),) * len(modules)
    if devices is not None:
        if len(modules) != len(devices):
            raise AssertionError(
                f'The number of modules {len(modules)} is not equal to the number of devices {len(devices)}' +
                pta_error(ErrCode.PARAM))
    else:
        devices = [None] * len(modules)
    devices = [_get_device_index(x, True) for x in devices]
    streams = [torch.npu.current_stream(x) for x in devices]
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

    def _worker(
        i: int,
        module: Module,
        input_t: Any,
        kwargs: Dict[str, Any],
        device: Optional[Union[int, torch.device]] = None,
        stream: Optional[torch.npu.Stream] = None,
    ) -> None:
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            t = get_a_var(input_t)
            if t is None:
                with lock:
                    results[i] = ExceptionWrapper(
                        where="in replica {}, no device was provided and no tensor input was found; "
                        "device cannot be resolved".format(i))
                return
            device = t.get_device()
        torch.npu.set_device(device)
        if stream is None:
            stream = torch.npu.current_stream(device)
        try:
            with torch.npu.device(device), torch.npu.stream(stream), autocast(enabled=autocast_enabled):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input_t, (list, tuple)):
                    input_t = (input_t,)
                output = module(*input_t, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = []
        for i, (module, input_t, kwargs, device, stream) in enumerate(zip(modules, inputs, kwargs_tup, devices, streams)):
            threads.append(threading.Thread(target=_worker, args=(i, module, input_t, kwargs, device, stream)))

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0], streams[0])

    outputs = []
    for i in range(len(inputs)):
        output = results.get(i)
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs


def npu_parallel_apply(self, replicas, inputs, kwargs) -> List[Any]:
    return _parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])


def npu_data_parallel(
    module: Module,
    inputs: Any,
    device_ids: Optional[Sequence[Union[int, torch.device]]] = None,
    output_device: Optional[Union[int, torch.device]] = None,
    dim: int = 0,
    module_kwargs: Optional[Any] = None,
) -> torch.Tensor:
    if not isinstance(inputs, tuple):
        inputs = (inputs,) if inputs is not None else ()

    device_type = _get_available_device_type()

    if device_type is None:
        raise RuntimeError("device type could not be determined" + pta_error(ErrCode.PARAM))

    if device_ids is None:
        device_ids = _get_all_device_indices()

    if device_ids is None:
        raise RuntimeError("no available devices were found" + pta_error(ErrCode.PARAM))

    if output_device is None:
        output_device = device_ids[0]

    device_ids = [_get_device_index(x, True) for x in device_ids]
    output_device = _get_device_index(output_device, True)
    src_device_obj = torch.device(device_type, device_ids[0])

    for t in chain(module.parameters(), module.buffers()):
        if t.device != src_device_obj:
            raise RuntimeError("module must have its parameters and buffers "
                               "on device {} (device_ids[0]) but found one of "
                               "them on device: {}".format(src_device_obj, t.device) + pta_error(ErrCode.VALUE))

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)

    if not inputs and not module_kwargs:
        inputs = ((),)
        module_kwargs = ({},)

    if module_kwargs is None:
        raise AssertionError(f'The module_kwargs is None' + pta_error(ErrCode.VALUE))

    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = _parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)


def _apply_module_patch():
    torch.nn.Module.npu = npu
    torch.nn.Module.to = to
    torch.nn.Module.cast_weight = cast_weight
    torch.nn.modules.rnn.LSTM.forward = _lstm_forward
    torch.nn.modules.batchnorm.SyncBatchNorm.forward = _syncbn_forward
    torch.utils.data.dataloader._MultiProcessingDataLoaderIter.__init__ = _mpdl_iter_init
    torch.nn.parallel.DataParallel.parallel_apply = npu_parallel_apply
    torch.nn.parallel.data_parallel = npu_data_parallel
