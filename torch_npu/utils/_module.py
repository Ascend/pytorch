from __future__ import annotations

import os
import sys
import warnings

import torch
import torch.distributed as pytorch_dist
from torch.distributed import Reducer
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter

import torch_npu
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
    if device is not None:
        device = torch.device("npu", device)
    else:
        device = torch.device("npu")
    if torch_npu.npu.is_available():
        with torch.no_grad():
            self.cast_weight(device)
    return self._apply(lambda t: t.npu(device))


def to(self, *args, **kwargs):
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
        *args, **kwargs
    )

    if dtype is not None:
        if not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError(
                "nn.Module.to only accepts floating point or complex "
                f"dtypes, but got desired dtype={dtype}" + pta_error(ErrCode.TYPE)
            )
        if dtype.is_complex:
            warnings.warn(
                "Complex modules are a new feature under active development whose design may change, "
                "and some modules might not work as expected when using complex tensors as parameters or buffers. "
            )
    if torch_npu.npu.is_available():
        with torch.no_grad():
            self.cast_weight(device)

    def convert(t):
        if convert_to_format is not None and t.dim() == 4:
            return t.to(
                device,
                dtype if t.is_floating_point() or t.is_complex() else None,
                non_blocking,
                memory_format=convert_to_format,
            )
        return t.to(
            device,
            dtype if t.is_floating_point() or t.is_complex() else None,
            non_blocking,
        )

    return self._apply(convert)


def cast_weight(self, device):
    def _format_cast(module, class_name):
        if (
            issubclass(class_name, torch.nn.Linear)
            and not torch.npu.get_mm_bmm_format_nd()
        ):
            module.weight.data = module.weight.data.to(device)
            module.weight.data = torch_npu.npu_format_cast(
                module.weight.data, 29
            )  # ACL_FORMAT_FRACTAL_NZ
        if (
            issubclass(class_name, torch.nn.MultiheadAttention)
            and module.q_proj_weight is not None
            and not torch.npu.get_mm_bmm_format_nd()
        ):
            module.q_proj_weight.data = module.q_proj_weight.data.to(device)
            module.q_proj_weight.data = torch_npu.npu_format_cast(
                module.q_proj_weight.data, 29
            )
            module.k_proj_weight.data = module.k_proj_weight.data.to(device)
            module.k_proj_weight.data = torch_npu.npu_format_cast(
                module.k_proj_weight.data, 29
            )
            module.v_proj_weight.data = module.v_proj_weight.data.to(device)
            module.v_proj_weight.data = torch_npu.npu_format_cast(
                module.v_proj_weight.data, 29
            )

        if torch.npu.is_jit_compile_false():
            return
        if issubclass(class_name, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            if module.affine:
                module.weight.data = module.weight.data.to(device)
                module.weight.data = torch_npu.npu_format_cast(
                    module.weight.data, 3
                )  # ACL_FORMAT_NC1HWC0
                module.bias.data = module.bias.data.to(device)
                module.bias.data = torch_npu.npu_format_cast(module.bias.data, 3)
            if module.track_running_stats:
                module.running_mean.data = module.running_mean.data.to(device)
                module.running_mean.data = torch_npu.npu_format_cast(
                    module.running_mean.data, 3
                )
                module.running_var.data = module.running_var.data.to(device)
                module.running_var.data = torch_npu.npu_format_cast(
                    module.running_var.data, 3
                )
        if issubclass(class_name, torch.nn.BatchNorm3d):
            # at present can not cast 1d to NDC1HWC0
            return
        if issubclass(class_name, torch.nn.Conv2d):
            if module.groups > 1:
                return
            if (
                hasattr(module, "weight")
                and module.weight is not None
                and "weight" in dict(module.named_parameters())
            ):
                module.weight.data = module.weight.data.to(device)
                module.weight.data = torch_npu.npu_format_cast(
                    module.weight.data, 4
                )  # ACL_FORMAT_FRACTAL_Z
        if issubclass(class_name, torch.nn.LazyConv3d):
            return
        if issubclass(class_name, torch.nn.Conv3d):
            module.weight.data = module.weight.data.to(device)
            device_name = torch_npu.npu.get_device_name()
            if any(
                device_name.startswith(prefix)
                for prefix in CONV3D_SUPPORT_FP32_SOC_PREFIX
            ):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 33)
                return
            module.weight.data = torch_npu.npu_format_cast(
                module.weight.data.half(), 33
            ).float()  # ACL_FRACTAL_Z_3D

    if device is None or "npu" not in str(device):
        return

    current_class = self.__class__
    _format_cast(self, current_class)

    if not self.children:
        return

    for sub_module in self.children():
        if isinstance(sub_module, torch.nn.Module):
            sub_module.cast_weight(device)


def _ddp_init_helper(
    self,
    parameters,
    expect_sparse_gradient,
    param_to_name_mapping,
    static_graph,
):
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
    (bucket_indices, per_bucket_size_limits) = (
        torch_npu.distributed._compute_bucket_assignment_by_size(
            parameters, bucket_size_limits, expect_sparse_gradient
        )
    )

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
        pytorch_dist._DEFAULT_FIRST_BUCKET_BYTES,
    )

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
    if os.getenv("ASCEND_RT_VISIBLE_DEVICES") == "":
        origin_mpdl_iter_init(self, *args, **kwargs)
        return

    try:
        torch_npu.npu.synchronize()
    except Exception as e:
        print(e)
    torch_npu._C._npu_set_thread_affinity(-1, -1)
    origin_mpdl_iter_init(self, *args, **kwargs)
    torch_npu._C._npu_reset_thread_affinity()


def _apply_module_patch():
    torch.nn.Module.npu = npu
    torch.nn.Module.to = to
    torch.nn.Module.cast_weight = cast_weight
    torch.utils.data.dataloader._MultiProcessingDataLoaderIter.__init__ = (
        _mpdl_iter_init
    )
