import warnings
from collections import defaultdict
import collections.abc as container_abcs
from typing import List

import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator, OptState, \
    _refresh_per_optimizer_state
from torch.cuda.amp.grad_scaler import GradScaler as Cuda_GradScaler
import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error
from .common import amp_definitely_not_available


class _NpuMultiDeviceReplicator(_MultiDeviceReplicator):
    """
    Lazily serves copies of a tensor to requested devices.  Copies are cached per-device.
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        if not master_tensor.is_npu:
            raise ValueError("Device type of master_tensor should be npu." + pta_error(ErrCode.VALUE))
        self.master = master_tensor
        self._per_device_tensors = {}


class GradScaler(Cuda_GradScaler):
    """
    An instance ``scaler`` of :class:`GradScaler` helps perform the steps of gradient scaling
    conveniently.

    * ``scaler.scale(loss)`` multiplies a given loss by ``scaler``'s current scale factor.
    * ``scaler.step(optimizer)`` safely unscales gradients and calls ``optimizer.step()``.
    * ``scaler.update()`` updates ``scaler``'s scale factor.

    Example::

        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()

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

    See the :ref:`Automatic Mixed Precision examples<amp-examples>` for usage
    (along with autocasting) in more complex cases like gradient clipping, gradient accumulation, gradient penalty,
    and multiple losses/optimizers.

    ``scaler`` dynamically estimates the scale factor each iteration.  To minimize gradient underflow,
    a large scale factor should be used.  However, ``float16`` values can "overflow" (become inf or NaN) if
    the scale factor is too large.  Therefore, the optimal scale factor is the largest factor that can be used
    without incurring inf or NaN gradient values.
    ``scaler`` approximates the optimal scale factor over time by checking the gradients for infs and NaNs during every
    ``scaler.step(optimizer)`` (or optional separate ``scaler.unscale_(optimizer)``, see :meth:`unscale_`).

    * If infs/NaNs are found, ``scaler.step(optimizer)`` skips the underlying ``optimizer.step()`` (so the params
      themselves remain uncorrupted) and ``update()`` multiplies the scale by ``backoff_factor``.

    * If no infs/NaNs are found, ``scaler.step(optimizer)`` runs the underlying ``optimizer.step()`` as usual.
      If ``growth_interval`` unskipped iterations occur consecutively, ``update()`` multiplies the scale by
      ``growth_factor``.

    The scale factor often causes infs/NaNs to appear in gradients for the first few iterations as its
    value calibrates.  ``scaler.step`` will skip the underlying ``optimizer.step()`` for these
    iterations.  After that, step skipping should occur rarely (once every few hundred or thousand iterations).

    Args:
        init_scale (float, optional, default=2.**16):  Initial scale factor.
        growth_factor (float, optional, default=2.0):  Factor by which the scale is multiplied during
            :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
        backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
            :meth:`update` if inf/NaN gradients occur in an iteration.
        growth_interval (int, optional, default=2000):  Number of consecutive iterations without inf/NaN gradients
            that must occur for the scale to be multiplied by ``growth_factor``.
        dynamic (bool, optional, default=True):  If ``False``, use static loss scale.
        enabled (bool, optional, default=True):  If ``False``, disables gradient scaling. :meth:`step` simply
            invokes the underlying ``optimizer.step()``, and other methods become no-ops.
    """

    def __init__(self,
                 init_scale=2. ** 16,
                 growth_factor=2.0,
                 backoff_factor=0.5,
                 growth_interval=2000,
                 dynamic=True,
                 enabled=True):
        if enabled and amp_definitely_not_available():
            warnings.warn("torch_npu.amp.GradScaler is enabled, but NPU is not available.  Disabling.")
            self._enabled = False
        else:
            self._enabled = enabled

        if self._enabled:
            if growth_factor <= 1.0:
                raise ValueError("The growth factor must be > 1.0." + pta_error(ErrCode.VALUE))
            if backoff_factor >= 1.0:
                raise ValueError("The backoff factor must be < 1.0." + pta_error(ErrCode.VALUE))

            self._init_scale = init_scale
            # self._scale will be lazily initialized during the first call to scale()
            self._scale = None
            self._growth_factor = growth_factor
            self._backoff_factor = backoff_factor
            self._growth_interval = growth_interval
            self._dynamic = dynamic
            self._init_growth_tracker = 0
            # self._growth_tracker will be lazily initialized during the first call to scale()
            self._growth_tracker = None
            self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
            self._has_overflow = False
            self._clear_overflow_flag = False
            self._dist_initialized = False
            self._dist_overflow_count = None

    def _lazy_init_scale_growth_tracker(self, dev):
        if self._growth_tracker is not None:
            raise RuntimeError("_growth_tracker initialized before _scale" + pta_error(ErrCode.VALUE))

        self._scale = torch.full((), self._init_scale, dtype=torch.float32).pin_memory().to(dev, non_blocking=True)
        self._growth_tracker = torch.full((), self._init_growth_tracker, dtype=torch.int32)
        self._growth_tracker = self._growth_tracker.pin_memory().to(dev, non_blocking=True)

    def _lazy_init_dist_flag_and_dist_overflow_count(self):
        if self._dist_overflow_count is not None:
            raise RuntimeError("_dist_overflow_count initialized before _scale" + pta_error(ErrCode.VALUE))
        try:
            if dist.is_initialized():
                self._dist_initialized = True
        except AttributeError as err:
            print("torch.distributed has no attribute is_initialized")

        self._dist_overflow_count = torch.Tensor([0.]).to('npu')

    def scale(self, outputs):
        """
        Multiplies ('scales') a tensor or list of tensors by the scale factor.

        Returns scaled outputs.  If this instance of :class:`GradScaler` is not enabled, outputs are returned
        unmodified.

        Args:
            outputs (Tensor or iterable of Tensors):  Outputs to scale.
        """
        if not self._enabled:
            return outputs

        if self._dist_overflow_count is None:
            self._lazy_init_dist_flag_and_dist_overflow_count()
            if self._dist_overflow_count is None:
                raise RuntimeError("_dist_overflow_count is None." + pta_error(ErrCode.VALUE))

        if self._dynamic and not self._clear_overflow_flag:
            if not torch_npu.npu.utils.is_support_inf_nan():
                GradScaler.clear_npu_overflow_flag()
            self._clear_overflow_flag = True

        # Short-circuit for the common case.
        if isinstance(outputs, torch.Tensor):
            if not outputs.is_npu:
                raise ValueError("Device type of outputs should be npu." + pta_error(ErrCode.VALUE))
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            if self._scale is None:
                raise RuntimeError("_scale is None." + pta_error(ErrCode.VALUE))
            return outputs * self._scale.to(device=outputs.device, non_blocking=True)

        # Invoke the more complex machinery only if we're treating multiple outputs.
        stash: List[_NpuMultiDeviceReplicator] = []  # holds a reference that can be overwritten by apply_scale

        def apply_scale(val):
            if isinstance(val, torch.Tensor):
                if not val.is_npu:
                    raise ValueError("Device type of val should be npu." + pta_error(ErrCode.VALUE))
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    if self._scale is None:
                        raise RuntimeError("_scale is None." + pta_error(ErrCode.VALUE))
                    stash.append(_NpuMultiDeviceReplicator(self._scale))
                return val * stash[0].get(val.device)
            elif isinstance(val, container_abcs.Iterable):
                iterable = map(apply_scale, val)
                if isinstance(val, list) or isinstance(val, tuple):
                    return type(val)(iterable)
                else:
                    return iterable
            else:
                raise TypeError("outputs must be a Tensor or an iterable of Tensors" + pta_error(ErrCode.TYPE))

        return apply_scale(outputs)

    def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16):
        per_device_found_inf = _NpuMultiDeviceReplicator(found_inf)
        per_device_inv_scale = _NpuMultiDeviceReplicator(inv_scale)

        # To set up _amp_foreach_non_finite_check_and_unscale_, split grads by device and dtype.
        # There could be hundreds of grads, so we'd like to iterate through them just once.
        # However, we don't know their devices or dtypes in advance.

        # Google says mypy struggles with defaultdicts type annotations.
        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        with torch.no_grad():
            if hasattr(optimizer, 'is_fused_optimizer'):
                if not optimizer.is_params_grads_combined:
                    optimizer._maybe_init_combined_params_and_grads()

                device = found_inf.device
                for grads_combined_one_dtype in optimizer.grads_all_group_combined:
                    if grads_combined_one_dtype is None:
                        continue
                    if self._dynamic:
                        torch._amp_foreach_non_finite_check_and_unscale_(
                            [grads_combined_one_dtype],
                            per_device_found_inf.get(device),
                            per_device_inv_scale.get(device))
                        if per_device_found_inf.get(device).item() > 0:
                            self._has_overflow = True
                    else:
                        grads_combined_one_dtype.mul_(
                            per_device_inv_scale.get(device))
            else:
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is None:
                            continue
                        if (not allow_fp16) and param.grad.dtype == torch.float16:
                            raise TypeError("Attempting to unscale FP16 gradients." + pta_error(ErrCode.TYPE))
                        if param.grad.is_sparse:
                            # is_coalesced() == False means the sparse grad has values with duplicate indices.
                            # coalesce() deduplicates indices and adds all values that have the same index.
                            # For scaled fp16 values, there's a good chance coalescing will cause overflow,
                            # so we should check the coalesced _values().
                            if param.grad.dtype == torch.float16:
                                param.grad = param.grad.coalesce()
                            to_unscale = param.grad._values()
                        else:
                            to_unscale = param.grad

                        per_device_and_dtype_grads[to_unscale.device][to_unscale.dtype].append(to_unscale)

                for device, per_dtype_grads in per_device_and_dtype_grads.items():
                    for grads in per_dtype_grads.values():
                        if self._dynamic:
                            torch._amp_foreach_non_finite_check_and_unscale_(grads,
                                                                             per_device_found_inf.get(device),
                                                                             per_device_inv_scale.get(device))
                            if per_device_found_inf.get(device).item() > 0:
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

    def unscale_(self, optimizer):
        """
        Divides ("unscales") the optimizer's gradient tensors by the scale factor.

        :meth:`unscale_` is optional, serving cases where you need to
        :ref:`modify or inspect gradients<working-with-unscaled-gradients>`
        between the backward pass(es) and :meth:`step`.
        If :meth:`unscale_` is not called explicitly,  gradients will be unscaled  automatically during :meth:`step`.

        Simple example, using :meth:`unscale_` to enable clipping of unscaled gradients::

            ...
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

        Args:
            optimizer (torch.optim.Optimizer):  Optimizer that owns the gradients to be unscaled.

        .. note::
            :meth:`unscale_` does not incur a CPU-NPU sync.

        .. warning::
            :meth:`unscale_` should only be called once per optimizer per :meth:`step` call,
            and only after all gradients for that optimizer's assigned parameters have been accumulated.
            Calling :meth:`unscale_` twice for a given optimizer between each :meth:`step` triggers a RuntimeError.

        .. warning::
            :meth:`unscale_` may unscale sparse gradients out of place, replacing the ``.grad`` attribute.
        """
        if not self._enabled:
            return

        self._check_scale_growth_tracker("unscale_")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update()." +
                               pta_error(ErrCode.INTERNAL))
        elif optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step()." +
                               pta_error(ErrCode.INTERNAL))

        # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
        if self._scale is None:
            raise RuntimeError("_scale is None." + pta_error(ErrCode.VALUE))
        inv_scale = self._scale.float().reciprocal()
        found_inf = torch.full((), 0.0, dtype=torch.float32).pin_memory().to(self._scale.device, non_blocking=True)

        optimizer_state["found_inf_per_device"] = self._unscale_grads_(optimizer, inv_scale, found_inf, False)
        optimizer_state["stage"] = OptState.UNSCALED

    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        retval = None
        if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()) and not self._has_overflow:
            retval = optimizer.step(*args, **kwargs)
        else:
            print("Gradient overflow. Skipping step")
        return retval

    def step(self, optimizer, *args, **kwargs):
        """
        :meth:`step` carries out the following two operations:

        1.  Internally invokes ``unscale_(optimizer)`` (unless :meth:`unscale_` was explicitly called for ``optimizer``
            earlier in the iteration).  As part of the :meth:`unscale_`, gradients are checked for infs/NaNs.
        2.  If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the unscaled
            gradients.  Otherwise, ``optimizer.step()`` is skipped to avoid corrupting the params.

        ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.

        Returns the return value of ``optimizer.step(*args, **kwargs)``.

        Args:
            optimizer (torch.optim.Optimizer):  Optimizer that applies the gradients.
            args:  Any arguments.
            kwargs:  Any keyword arguments.

        .. warning::
            Closure use is not currently supported.
        """
        if (not self._enabled):
            return optimizer.step(*args, **kwargs)

        if "closure" in kwargs:
            raise RuntimeError("Closure use is not currently supported if GradScaler is enabled." +
                               pta_error(ErrCode.NOT_SUPPORT))

        self._check_scale_growth_tracker("step")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("step() has already been called since the last update()." +
                               pta_error(ErrCode.INTERNAL))

        retval = None

        if (hasattr(optimizer, "_step_supports_amp_scaling") and optimizer._step_supports_amp_scaling):
            # This optimizer has customized scale-handling logic, so we can call optimizer.step() directly.
            # The contract with custom optimizers is that their step() should accept an additional,
            # optional grad_scaler kwarg.  We append self to the kwargs so the custom optimizer has full information:
            # it can query its own state, invoke unscale_ on itself, etc
            retval = optimizer.step(*args, **dict(kwargs, grad_scaler=self))
            optimizer_state["stage"] = OptState.STEPPED
            return retval

        if optimizer_state["stage"] is OptState.READY:
            self.unscale_(optimizer)

        if len(optimizer_state["found_inf_per_device"]) <= 0:
            raise RuntimeError("No inf checks were recorded for this optimizer." + pta_error(ErrCode.INTERNAL))

        if self._dynamic:
            retval = self._maybe_opt_step(optimizer, optimizer_state, *args, **kwargs)
            optimizer_state["stage"] = OptState.STEPPED
            return retval

        retval = optimizer.step(*args, **kwargs)
        optimizer_state["stage"] = OptState.STEPPED

        return retval

    def update(self, new_scale=None):
        """
        Updates the scale factor.

        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.

        Passing ``new_scale`` sets the scale directly.

        Args:
            new_scale (float or :class:`torch.npu.FloatTensor`, optional, default=None):  New scale factor.

        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.
        """
        if not self._enabled:
            return

        _scale, _ = self._check_scale_growth_tracker("update")

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale = torch.full((), new_scale, dtype=torch.float32)
                self._scale = self._scale.pin_memory().to(_scale.device, non_blocking=True)
            else:
                reason = "new_scale should be a float or a 1-element torch.npu.FloatTensor with requires_grad=False."
                if not isinstance(new_scale, torch.npu.FloatTensor):  # type: ignore[attr-defined]
                    raise TypeError(reason + pta_error(ErrCode.TYPE))
                if new_scale.numel() != 1:
                    raise ValueError(reason + pta_error(ErrCode.VALUE))
                if new_scale.requires_grad:
                    raise ValueError(reason + pta_error(ErrCode.VALUE))
                self._scale = new_scale
        elif self._dynamic:
            self._npu_update_scale()

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
        self._has_overflow = False
        self._clear_overflow_flag = False

    def state_dict(self):
        state = super(GradScaler, self).state_dict()
        if self._enabled:
            state["dynamic"] = self._dynamic
        return state

    def load_state_dict(self, state_dict):
        if not self._enabled:
            return

        if len(state_dict) == 0:
            raise RuntimeError("The source state dict is empty, possibly because it was saved "
                               "from a disabled instance of GradScaler." + pta_error(ErrCode.VALUE))

        super(GradScaler, self).load_state_dict(state_dict)
        self._dynamic = state_dict["dynamic"]

    @staticmethod
    def get_npu_overflow_flag():
        float_status = torch.zeros(8).pin_memory().to('npu', non_blocking=True)
        result = torch_npu.npu_get_float_status(float_status)
        if (result.cpu()[0] != 0):
            return True
        else:
            return False

    @staticmethod
    def clear_npu_overflow_flag():
        float_status = torch.zeros(8).pin_memory().to('npu', non_blocking=True)
        result = torch_npu.npu_clear_float_status(float_status)

    def _sync_dist_overflow_count(self):
        if torch_npu.npu.utils.is_support_inf_nan():
            return
        if self._dynamic and self._dist_initialized:
            if self._has_overflow:
                self._dist_overflow_count.add_(1)
                dist.all_reduce(self._dist_overflow_count)
                self._dist_overflow_count.zero_()
            else:
                dist.all_reduce(self._dist_overflow_count)
                if self._dist_overflow_count.item() != 0:
                    self._has_overflow = True
                self._dist_overflow_count.zero_()

    def _npu_update_scale(self):
        if self._has_overflow:
            self._scale.mul_(self._backoff_factor)
            self._growth_tracker.zero_()
            print(("Loss scaler reducing loss scale "
                   "to {}").format(self._scale.item()))
        else:
            # Entering this branch means we just carried out a successful step,
            # so growth_tracker is incremented before comparing to growth_interval.
            self._growth_tracker.add_(1)
            if self._growth_tracker.item() == self._growth_interval:
                new_scale = self._scale * self._growth_factor
                if not torch.isinf(new_scale):
                    self._scale = new_scale
                self._growth_tracker.zero_()
                print(("Loss scaler increasing loss scale "
                       "to {}").format(self._scale.item()))
