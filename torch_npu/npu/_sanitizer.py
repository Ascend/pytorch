# Copyright (c) 2026, Huawei Technologies Co., Ltd
"""torch_npu NPU sanitizer: the record_stream/free-time mount plus the
engine and distributed observation layers that feed it.

The judgment machinery lives in ``_stream_check.py`` and is untouched:
the observation layers only submit pseudo kernel launches — empty
read/write sets plus the accessed storage pointers — so usages no
Python-dispatched op reveals (engine backwards, eager collectives)
enter the free-time ledger without adding data-race noise.
"""
import os
import atexit
import functools
import logging
import types
import warnings

import torch
import torch_npu
import torch.distributed as dist
from torch.autograd import Variable
import torch_npu.utils._npu_trace as npu_trace
import torch_npu.npu._stream_check as stream_check
import torch_npu.npu._kernel_check as kernel_check
from torch_npu.utils.utils import _print_warn_log

logger = logging.getLogger(__name__)


class SanitizerMode:
    STREAM = 0
    KERNEL = 1


_P2POP_CLS = getattr(dist, "P2POp", None)


def _flatten_ptrs(value, out):
    if isinstance(value, torch.Tensor):
        if not value.is_npu:
            return
        try:
            storage = value.untyped_storage()
            ptr = storage.data_ptr() if storage is not None else 0
        except (RuntimeError, AttributeError):
            return
        if ptr:
            out.add(ptr)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _flatten_ptrs(item, out)
    elif _P2POP_CLS is not None and isinstance(value, _P2POP_CLS):
        _flatten_ptrs(value.tensor, out)


def _contains_any_tensor(value):
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_any_tensor(item) for item in value)
    tensor = getattr(value, "tensor", None)
    return isinstance(tensor, torch.Tensor)


def _record(stream, ptrs, tag, handler):
    if handler is None or stream is None or not ptrs:
        return
    handler._handle_kernel_launch(
        stream, set(), set(), set(), tag, {},
        storage_dataptrs_accessed=ptrs,
    )


class _EngineProxy:
    """Forwards everything except ``run_backward`` to the real engine."""

    def __init__(self, engine):
        self._orig = engine
        self.handler = None

    def __getattr__(self, name):
        return getattr(self._orig, name)

    def run_backward(self, *args, **kwargs):
        handler = self.handler
        if handler is None:
            return self._orig.run_backward(*args, **kwargs)
        try:
            stream = int(torch_npu.npu.current_stream().npu_stream)
        except (RuntimeError, AttributeError):
            stream = None
        tensors = args[0] if args else kwargs.get("tensors")
        grads = args[1] if len(args) > 1 else kwargs.get("grad_tensors")
        observation = _engine_observation
        observation.self_check(args, kwargs, tensors, grads)
        observation.record(stream, tensors, handler)
        observation.record(stream, grads, handler)
        ret = self._orig.run_backward(*args, **kwargs)
        out_grads = [t for t in (ret or ()) if t is not None]
        if not out_grads:
            out_grads = [
                t.grad
                for t in (tensors or ())
                if isinstance(t, torch.Tensor)
                and t.is_leaf
                and t.grad is not None
            ]
        observation.record(stream, out_grads, handler)
        return ret


class _EngineObservation:
    """Feed the autograd engine's ``run_backward`` boundary tensors to
    the upstream free-time judgment through a proxy installed on
    ``Variable._execution_engine``; records go through the shared
    pseudo kernel launch and are tagged ``"autograd_engine [engine]"``.

    Known shapes, pinned by tests: with ``inputs=()`` non-leaf subgraph
    outputs the engine offers no return channel, so the exit side
    records nothing; the ``run_graph_backward`` + ``resize_(0)``
    release shape reports by design — that code relies on
    synchronization before reuse, not before free.
    """

    OPERATOR = "autograd_engine [engine]"

    def __init__(self):
        self.installed = False
        self.orig_engine = None
        self.proxy = None
        self.extraction_warned = False

    def record(self, stream, tensors, handler):
        ptrs = set()
        _flatten_ptrs(tensors, ptrs)
        _record(stream, ptrs, self.OPERATOR, handler)

    def _call_plausible(self, args, kwargs, tensors, grads):
        """Whether the boundary objects match a known engine calling
        shape (leading positional tensors or the ``tensors``/
        ``grad_tensors`` keywords); only a drifted protocol does not."""
        if args:
            return _contains_any_tensor(tensors) or _contains_any_tensor(grads)
        if kwargs:
            return "tensors" in kwargs or "grad_tensors" in kwargs
        return True

    def self_check(self, args, kwargs, tensors, grads):
        if self.extraction_warned:
            return
        self.extraction_warned = True
        if self._call_plausible(args, kwargs, tensors, grads):
            return
        warnings.warn(
            "torch_npu sanitizer: no tensor found in the autograd engine call "
            "and none of the expected argument names is present (the "
            "run_backward calling protocol may have changed); backward "
            "observation is degraded",
            stacklevel=3,
        )

    def install(self, handler):
        """Install the proxy idempotently: a live proxy is only
        rebound, and a proxy restored by a third-party layer is
        reused, not rewrapped."""
        engine = Variable._execution_engine
        if self.proxy is not None and engine is self.proxy:
            self.proxy.handler = handler
            return
        if isinstance(engine, _EngineProxy):
            proxy = engine
            self.orig_engine = proxy._orig
        else:
            self.orig_engine = engine
            proxy = _EngineProxy(engine)
            Variable._execution_engine = proxy
        if self.proxy is not None and self.proxy is not proxy:
            self.proxy.handler = None
        proxy.handler = handler
        self.proxy = proxy
        self.installed = True

    def uninstall(self):
        """Restore the original engine; a slot taken over by a
        third-party layer is left alone (its proxy is inert)."""
        if self.installed:
            if self.proxy is not None:
                self.proxy.handler = None
                if Variable._execution_engine is self.proxy:
                    Variable._execution_engine = self.proxy._orig
            self.proxy = None
            self.installed = False
            self.orig_engine = None


_engine_observation = _EngineObservation()


class _DistObservation:
    """Wrap the eager ``torch.distributed`` entry functions and record
    each call's input/output tensors on the caller's stream through
    the shared pseudo kernel launch, tagged ``"dist [<function>]"``.

    Operands the runtime already protected are skipped: the wrapper
    snapshots ``recorded_streams`` around the call and drops what
    ProcessGroupHCCL recorded through the allocator during it.

    ``isend``/``irecv`` are deliberately unwrapped — ``P2POp``
    validates its ``op`` by identity, so wrapping them would break
    ``P2POp(dist.isend, ...)``. ``batch_isend_irecv`` is wrapped
    instead and records every ``P2POp.tensor``; direct ``isend``/
    ``irecv`` calls stay unobserved (documented blind spot).

    Wrappers carry a marker, so an install after a third-party
    save-replace-restore patch reuses the restored wrapper instead of
    double-wrapping. Protocol drift and spec entries missing from the
    current torch degrade with one-time ``UserWarning``s.
    """

    WRAPPER_TAG = "__torch_npu_sanitizer_dist_wrapper__"

    # name -> (input positions, output positions, input kw names, output kw names)
    SPECS = {
        "all_reduce": ((0,), (0,), ("tensor",), ("tensor",)),
        "broadcast": ((0,), (0,), ("tensor",), ("tensor",)),
        "all_gather": ((1,), (0,), ("tensor",), ("tensor_list",)),
        "_all_gather_base": ((1,), (0,), ("input_tensor",), ("output_tensor",)),
        "all_gather_into_tensor": ((1,), (0,), ("input_tensor",), ("output_tensor",)),
        "all_gather_single": ((1,), (0,), ("input_tensor",), ("output_tensor",)),
        "reduce_scatter": ((1,), (0,), ("input_list",), ("output",)),
        "_reduce_scatter_base": ((1,), (0,), ("input",), ("output",)),
        "reduce_scatter_tensor": ((1,), (0,), ("input",), ("output",)),
        "reduce_scatter_single": ((1,), (0,), ("input",), ("output",)),
        "all_to_all": ((1,), (0,), ("input_tensor_list",), ("output_tensor_list",)),
        "all_to_all_single": ((1,), (0,), ("input",), ("output",)),
        "gather": ((0,), (1,), ("tensor",), ("gather_list",)),
        "scatter": ((1,), (0,), ("scatter_list",), ("tensor",)),
        "reduce": ((0,), (0,), ("tensor",), ("tensor",)),
        "batch_isend_irecv": ((0,), (), ("p2p_op_list",), ()),
        "send": ((0,), (), ("tensor",), ()),
        "recv": ((0,), (), ("tensor",), ()),
    }

    def __init__(self):
        self.installed = False
        self.origs = {}
        self.wrappers = {}
        self.warned = set()

    def _collect_args(self, args, kwargs, positions, names):
        out = []
        for pos, name in zip(positions, names):
            item = args[pos] if pos < len(args) else kwargs.get(name)
            if isinstance(item, (list, tuple)):
                out.extend(item)
            elif item is not None:
                out.append(item)
        return out

    def _call_plausible(self, args, kwargs, positions, names):
        if args:
            return any(
                pos < len(args) and _contains_any_tensor(args[pos])
                for pos in positions
            )
        if kwargs:
            return any(name in kwargs for name in names)
        return True

    def self_check(self, name, args, kwargs, positions, names):
        if name in self.warned:
            return
        self.warned.add(name)
        if self._call_plausible(args, kwargs, positions, names):
            return
        warnings.warn(
            f"torch_npu sanitizer: no tensor found in the '{name}' "
            "torch.distributed call and none of the expected argument "
            "names is present (the calling protocol may have changed); "
            "communication observation is degraded",
            stacklevel=3,
        )

    def warn_missing_entry(self, name):
        if name in self.warned:
            return
        self.warned.add(name)
        warnings.warn(
            f"torch_npu sanitizer: torch.distributed has no entry '{name}' "
            "on this torch version (removed or renamed?); communication "
            "observation for it is unavailable",
            stacklevel=3,
        )

    def _make_wrapper(self, name, orig, spec):
        in_pos, out_pos, in_names, out_names = spec
        if len(in_pos) != len(in_names) or len(out_pos) != len(out_names):
            raise ValueError(
                f"torch_npu sanitizer: malformed SPECS entry '{name}': "
                "position and keyword tuples must have equal length"
            )
        positions = in_pos + out_pos
        names = in_names + out_names
        tag = f"dist [{name}]"

        def wrapper(*args, **kwargs):
            handler = wrapper.handler
            if handler is None:
                return orig(*args, **kwargs)
            self.self_check(name, args, kwargs, positions, names)
            tensors = (
                self._collect_args(args, kwargs, in_pos, in_names)
                + self._collect_args(args, kwargs, out_pos, out_names)
            )
            ptrs = set()
            _flatten_ptrs(tensors, ptrs)
            recorded = {
                ptr: frozenset(handler._npu_tensors[ptr].recorded_streams)
                for ptr in ptrs
                if ptr in handler._npu_tensors
            }
            ret = orig(*args, **kwargs)
            try:
                stream = int(torch_npu.npu.current_stream().npu_stream)
            except (RuntimeError, AttributeError):
                return ret
            compensated = {
                ptr
                for ptr in ptrs
                if ptr in handler._npu_tensors
                and frozenset(handler._npu_tensors[ptr].recorded_streams)
                != recorded.get(ptr)
            }
            _record(stream, ptrs - compensated, tag, handler)
            return ret

        wrapper = functools.wraps(orig)(wrapper)
        wrapper.handler = None
        wrapper.__dict__[self.WRAPPER_TAG] = orig
        return wrapper

    def install(self, handler):
        """Wrap every spec entry idempotently: live wrappers are only
        rebound, wrappers restored by third-party patches are reused,
        entries missing on this torch warn once, and the wrapped
        count is logged."""
        wrapped = 0
        for name, spec in self.SPECS.items():
            current = getattr(dist, name, None)
            if current is None:
                self.warn_missing_entry(name)
                continue
            if self.wrappers.get(name) is current:
                wrapper = current
            elif isinstance(current, types.FunctionType) and \
                    current.__dict__.get(self.WRAPPER_TAG) is not None:
                wrapper = current
                self.origs[name] = wrapper.__dict__[self.WRAPPER_TAG]
            else:
                wrapper = self._make_wrapper(name, current, spec)
                self.origs[name] = current
                setattr(dist, name, wrapper)
            displaced = self.wrappers.get(name)
            if displaced is not None and displaced is not wrapper:
                displaced.handler = None
            wrapper.handler = handler
            self.wrappers[name] = wrapper
            wrapped += 1
        self.installed = True
        logger.info(
            "torch_npu sanitizer: observing %d torch.distributed entry "
            "functions (%d in the spec table, %d missing on this torch)",
            wrapped, len(self.SPECS), len(self.SPECS) - wrapped,
        )

    def uninstall(self):
        """Restore the module functions still holding our wrappers;
        slots taken over by a third-party patch are left alone (their
        wrappers are inert)."""
        for name, wrapper in self.wrappers.items():
            wrapper.handler = None
            if getattr(dist, name, None) is wrapper:
                setattr(dist, name, self.origs[name])
        self.origs.clear()
        self.wrappers.clear()
        self.installed = False


_dist_observation = _DistObservation()


class NPUSanitizer:

    def __init__(self):
        self.event_handler = None
        self.dispatch = None
        self.kernel_path_manager = None
        self.mode = SanitizerMode.STREAM
        self.opp_debug_path = os.path.join(os.getcwd(), "opp_debug_path")
        self.opp_debug_kernel_path = os.getenv('ASCEND_OPP_DEBUG_PATH')
        self.enabled = False
        # record_stream detection is enabled by TORCH_NPU_SANITIZER=1
        self.check_record_stream = os.getenv('TORCH_NPU_SANITIZER', '0') == '1'

    def enable(self):
        if self.enabled:
            return
        if self.opp_debug_kernel_path:
            success = self.enable_kernel_check()
            self.mode = SanitizerMode.KERNEL
        else:
            success = self.enable_stream_check()
            self.mode = SanitizerMode.STREAM
        if not self.enabled and success:
            torch_npu._C._activate_npu_trace(self.mode)
            self.enabled = True

    def enable_kernel_check(self) -> bool:
        if not self.opp_debug_kernel_path:
            _print_warn_log("ASCEND_OPP_DEBUG_PATH is not set! TORCH_NPU_SANITIZER takes no effect!")
            return False
        self.kernel_path_manager = kernel_check.KernelPathManager()
        if not os.path.exists(self.opp_debug_path):
            return False
        self.event_handler = kernel_check.EventHandler()
        npu_trace.register_callback_for_acl_start_execution(
            self.event_handler._handle_acl_start_execution,
            "handle_acl_start_execution"
        )
        npu_trace.register_callback_for_acl_finish_execution(
            self.event_handler._handle_acl_finish_execution,
            "handle_acl_finish_execution"
        )
        return True

    def enable_stream_check(self) -> bool:
        # Use our extended NPURecordStreamHandler which inherits from PyTorch's EventHandler
        self.event_handler = stream_check.NPURecordStreamHandler()
        self.dispatch = stream_check.NPUSanitizerDispatchMode(self.event_handler)
        self.dispatch.__enter__()
        _engine_observation.install(self.event_handler)
        _dist_observation.install(self.event_handler)
        _register_atexit_flush_once()
        npu_trace.register_callback_for_npu_event_creation(
            self.event_handler._handle_event_creation,
            "handle_event_creation"
        )
        npu_trace.register_callback_for_npu_event_deletion(
            self.event_handler._handle_event_deletion,
            "handle_event_deletion"
        )
        npu_trace.register_callback_for_npu_event_record(
            self.event_handler._handle_event_record,
            "handle_event_record"
        )
        npu_trace.register_callback_for_npu_event_wait(
            self.event_handler._handle_event_wait,
            "handle_event_wait"
        )
        npu_trace.register_callback_for_npu_memory_allocation(
            self.event_handler._handle_memory_allocation,
            "handle_memory_allocation"
        )
        npu_trace.register_callback_for_npu_memory_deallocation(
            self.event_handler._handle_memory_deallocation,
            "handle_memory_deallocation"
        )
        npu_trace.register_callback_for_npu_stream_creation(
            self.event_handler._handle_stream_creation,
            "handle_stream_creation"
        )
        npu_trace.register_callback_for_npu_device_synchronization(
            self.event_handler._handle_device_synchronization,
            "handle_device_synchronization"
        )
        npu_trace.register_callback_for_npu_stream_synchronization(
            self.event_handler._handle_stream_synchronization,
            "handle_stream_synchronization"
        )
        npu_trace.register_callback_for_npu_event_synchronization(
            self.event_handler._handle_event_synchronization,
            "handle_event_synchronization"
        )
        npu_trace.register_callback_for_npu_record_stream(
            self.event_handler._handle_record_stream, "handle_record_stream"
        )
        npu_trace.register_callback_for_npu_erase_stream(
            self.event_handler._handle_erase_stream, "handle_erase_stream"
        )
        return True

    def __del__(self):
        try:
            if self.dispatch:
                self.dispatch.__exit__(None, None, None)
        except Exception:
            pass

    def clear_debug_env(self):
        if self.kernel_path_manager:
            self.kernel_path_manager.clear_debug_env()


def enable_npu_sanitizer():
    npu_sanitizer.enable()


npu_sanitizer = NPUSanitizer()

atexit.register(npu_sanitizer.clear_debug_env)

_atexit_flush_registered = False


def _flush_record_stream_warnings_atexit():
    """Flush pending record_stream reports through whichever handler is
    live when the process exits; a no-op when none is."""
    handler = npu_sanitizer.event_handler
    if handler is not None:
        handler.flush_record_stream_warnings()


def _register_atexit_flush_once():
    """Register the exit flush once per process: enable() has its own
    re-entrancy guard, but historical callers re-enabled freely, and a
    per-enable registration would pile up exit callbacks."""
    global _atexit_flush_registered
    if not _atexit_flush_registered:
        atexit.register(_flush_record_stream_warnings_atexit)
        _atexit_flush_registered = True
