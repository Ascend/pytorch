from collections import abc, defaultdict
import logging
from typing import Dict, List, Optional, Union

import torch
from torch.cuda import FloatTensor  # type: ignore[attr-defined]
from torch.distributed.distributed_c10d import ProcessGroup
import torch.distributed as dist
from torch.optim.sgd import SGD
import torch_npu
from torch_npu.npu.amp.grad_scaler import GradScaler, OptState, _NpuMultiDeviceReplicator
from torch_npu.utils.error_code import ErrCode, dist_error

logger = logging.getLogger(__name__)


def _refresh_per_optimizer_state():
    return {"stage": OptState.READY, "found_inf_per_device": {}}


def _is_supported_device(tensor: torch.Tensor):
    return tensor.is_npu or tensor.device.type in ("xla", "cpu")


class _GeneralMultiDeviceReplicator(_NpuMultiDeviceReplicator):
    """
    Lazily serves tensor to request device. This class extends
    _NpuMultiDeviceReplicator to allow support for "cpu" as a device.
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        if not _is_supported_device(master_tensor):
            raise RuntimeError("Device is not supported" + dist_error(ErrCode.NOT_SUPPORT))
        self.master = master_tensor
        self._per_device_tensors: Dict[torch.device, torch.Tensor] = {}


class ShardedGradScaler(GradScaler):
    """
    ShardedGradScaler helps perform gradient scaling in a shard aware manner. It extends
    functionality from GradScaler:
    * Suports Pytorch DDP and FSDP implementations
    * Support CPU offloaded tensors (as used in fully sharded data parallel[FSDP])
    * Supports the custom Mixed Precision loss dtype (fp16, bf16) that FSDP returns
    * Sync inf/nan for scaled gradient tensors on any torch.device (where tensors are placed) across
    nodes

    Example::

        # Creates a ShardedGradScaler once at the beginning of training.
        scaler = ShardedGradScaler()

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()

                # scaler.step() first unscales gradients of the optimizer's params.
                # If gradients don't contain infs/NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

    See :class:`GradScaler` for explanation of scaling/unscaling and more use cases.

    Args:
        init_scale (float, optional, default=2.**16):  Initial scale factor.
        growth_factor (float, optional, default=2.0):  Factor by which the scale is multiplied during
            :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
        backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
            :meth:`update` if inf/NaN gradients occur in an iteration.
        growth_interval (int, optional, default=2000):  Number of consecutive iterations without inf/NaN gradients
            that must occur for the scale to be multiplied by ``growth_factor``.
        enabled (bool, optional):  If ``False``, disables gradient scaling. :meth:`step` simply
            invokes the underlying ``optimizer.step()``, and other methods become no-ops.
            Default: ``True``
        process_group (ProcessGroup, optional, default=torch.distributed.group.WORLD):
            process group for sharding
    """

    def __init__(
            self,
            init_scale: float = 2.0 ** 16,
            backoff_factor: float = 0.5,
            growth_factor: float = 2.0,
            growth_interval: int = 2000,
            enabled: bool = True,
            process_group: Optional[ProcessGroup] = dist.group.WORLD,
            dynamic: bool = True
    ):
        super().__init__(
            init_scale=init_scale,
            backoff_factor=backoff_factor,
            growth_factor=growth_factor,
            growth_interval=growth_interval,
            enabled=enabled,
            dynamic=dynamic
        )
        if self._enabled:
            self.process_group = process_group
            self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def scale(self, outputs: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        if not self._enabled:
            return outputs

        if self._dist_overflow_count is None:
            self._lazy_init_dist_flag_and_dist_overflow_count()
            if self._dist_overflow_count is None:
                raise RuntimeError("Attribute _dist_overflow_count is abnormal" + dist_error(ErrCode.VALUE))
        if self._dynamic and not self._clear_overflow_flag:
            if not torch_npu.npu.utils.is_support_inf_nan():
                GradScaler.clear_npu_overflow_flag()
            self._clear_overflow_flag = True

        if isinstance(outputs, torch.Tensor):
            if not _is_supported_device(outputs):
                raise RuntimeError("Device is not supported" + dist_error(ErrCode.NOT_SUPPORT))
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            if self._scale is None:
                raise RuntimeError("Attribute _scale is abnormal" + dist_error(ErrCode.VALUE))
            scaled_output = outputs * self._scale.to(device=outputs.device, non_blocking=True)
            # Here we ensure the return dtype is the same as the outputs dtype.
            # For the FSDP + Mixed Precision use case, the loss output is in the Mixed Precision
            # format (fp16, bf16) and so the scaled loss should be of the same dtype.
            return scaled_output.type(outputs.dtype)

        stash: List[_GeneralMultiDeviceReplicator] = []

        def apply_scale(val: Union[torch.Tensor, abc.Iterable]) -> Union[torch.Tensor, abc.Iterable]:
            if isinstance(val, torch.Tensor):
                if not _is_supported_device(val):
                    raise RuntimeError("Device is not supported" + dist_error(ErrCode.NOT_SUPPORT))
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    if self._scale:
                        raise RuntimeError("Attribute _scale is abnormal" + dist_error(ErrCode.VALUE))
                    stash.append(_GeneralMultiDeviceReplicator(self._scale))
                scaled_val = val * stash[0].get(val.device)
                # Here we ensure the return dtype is the same as the outputs dtype.
                # For the FSDP + Mixed Precision use case, the loss output is in the Mixed Precision
                # format (fp16, bf16) and so the scaled loss should be of the same dtype.
                return scaled_val.type(val.dtype)
            elif isinstance(val, abc.Iterable):
                iterator = map(apply_scale, val)
                if isinstance(val, (list, tuple)):
                    return type(val)(iterator)
                else:
                    return iterator
            else:
                raise TypeError("outputs must be a Tensor or an iterable of Tensors" + dist_error(ErrCode.TYPE))

        return apply_scale(outputs)  # type: ignore[return-value]

    def _foreach_non_finite_check_and_unscale_cpu_(
            self, grads: List, found_inf: torch.Tensor, inv_scale: torch.Tensor
    ) -> None:
        if len(grads) == 0:
            return
        if inv_scale.numel() != 1:
            raise ValueError("inv_scale must be a 1-element tensor." + dist_error(ErrCode.VALUE))
        if found_inf.numel() != 1:
            raise ValueError("found_inf must be a 1-element tensor." + dist_error(ErrCode.VALUE))
        expected_device = grads[0].device
        for grad in grads:
            for tensor in grad:
                if tensor.device != expected_device:
                    logger.error("tensor device is %s and expected device is %s" % (tensor.device, expected_device))
                    raise ValueError("Gradients must be on the same device." + dist_error(ErrCode.VALUE))

                # check for non_overlapping_and_dense doesn't exist in the python world
                # we assume tensor is not MTA(multi tensor apply) safe. iterate through each item regardless of dtype
                if torch.isinf(tensor).any().item() is True or torch.isnan(tensor).any().item() is True:
                    found_inf.data = torch.tensor([1.0])
                    break
                else:
                    tensor.data *= inv_scale.item()

    def _unscale_grads_(
            self, optimizer: SGD, inv_scale: torch.Tensor, found_inf: torch.Tensor, allow_fp16: bool = True
    ) -> Dict[torch.device, torch.Tensor]:
        per_device_inv_scale = _GeneralMultiDeviceReplicator(inv_scale)
        per_device_found_inf = _GeneralMultiDeviceReplicator(found_inf)

        # To set up _amp_foreach_non_finite_check_and_unscale_, split grads by device and dtype.
        # There could be thousands of grads, so we'd like to iterate through them just once.
        # However, we don't know their devices or dtypes in advance.

        # Google says mypy struggles with defaultdicts type annotations.
        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))  # type: ignore[var-annotated]
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    if (not allow_fp16) and param.grad.dtype == torch.float16:
                        raise TypeError("Attempting to unscale FP16 gradients." + dist_error(ErrCode.TYPE))
                    if param.grad.is_sparse:
                        # is_coalesced() == False means the sparse grad has values with duplicate indices.
                        # coalesce() deduplicates indices and adds all values that have the same index.
                        # For scaled fp16 values, there's a good chance coalescing will cause overflow,
                        # so we should check the coalesced _values().
                        if param.grad.dtype is torch.float16:
                            # coalesce is not suported in torch.float16
                            param_grad_fp32 = param.grad.type(torch.float32).coalesce()
                            param.grad = param_grad_fp32.type(torch.float16)
                        to_unscale = param.grad._values()
                    else:
                        to_unscale = param.grad

                    per_device_and_dtype_grads[to_unscale.device][to_unscale.dtype].append(to_unscale)

            for device, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    if grads[0].device.type == "cpu":
                        self._foreach_non_finite_check_and_unscale_cpu_(
                            grads,
                            per_device_found_inf.get(device),
                            per_device_inv_scale.get(device),
                        )
                    else:
                        if self._dynamic:
                            torch._amp_foreach_non_finite_check_and_unscale_(grads,
                                                                             per_device_found_inf.get(device),
                                                                             per_device_inv_scale.get(device))
                            if per_device_found_inf.get(device)[0].item() > 0:
                                self._has_overflow = True
                        else:
                            for grad in grads:
                                grad.mul_(per_device_inv_scale.get(device))
            self._sync_dist_overflow_count()
            if self._has_overflow:
                per_device_found_inf.get(found_inf.device).add_(1)
            else:
                per_device_found_inf.get(found_inf.device)

        return per_device_found_inf._per_device_tensors

    def unscale_(self, optimizer: SGD) -> None:
        if not self._enabled:
            return

        self._check_scale_growth_tracker("unscale_")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update()." +
                               dist_error(ErrCode.INTERNAL))
        elif optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step()." + dist_error(ErrCode.INTERNAL))

        # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
        if self._scale is None:
            raise RuntimeError("Attribute _scale is abnormal" + dist_error(ErrCode.VALUE))
        inv_scale = self._scale.double().reciprocal().float()
        found_inf = torch.full((1,), 0.0, dtype=torch.float32, device=self._scale.device)

        optimizer_state["found_inf_per_device"] = self._unscale_grads_(optimizer, inv_scale, found_inf, True)
        optimizer_state["stage"] = OptState.UNSCALED

        # Synchronize the detected inf across the ranks
        optimizer_state = self._per_optimizer_states[id(optimizer)]

    def step(self, optimizer: SGD, *args, **kwargs) -> Optional[float]:
        return super().step(optimizer, *args, **kwargs)
