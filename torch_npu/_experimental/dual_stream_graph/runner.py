"""Opt-in dual-stream scheduling of two argument groups on isolated trees.

The runner stays outside the Dynamo entry: the caller splits the batch, hands
the argument groups to :meth:`DualBatchRunner.run`, and provides a ``recover``
callback that merges the per-lane outputs.  Each group runs to completion of
its whole subgraph chain (including eager bridges) inside its own
:class:`GraphExecutionDomain`, on that lane's own submit stream.

Safety invariants enforced here:

1. Graph-owned writable memory never escapes: every lane output tensor is
   cloned on the lane right after its full subgraph chain.
2. The caller stream waits for the previous request's completion event before
   buffers are reused, even when the caller stream changes between requests.
3. After recovery, a fresh completion event covers both lanes plus every read
   performed by ``recover``; input, lane-output and result storages are
   retained until that event, and results are registered on the caller stream
   via ``record_stream`` so post-return reads stay ordered.
4. A lane failure drains the device before Python references unwind, resets
   both domains, and never eagerly retries on possibly dirty buffers.

The runner never touches the process-wide manual step marker; native Dynamo
call generations delimit captures, one per lane.
"""

import threading

import torch
from torch.utils._pytree import tree_flatten, tree_map

from torch_npu._experimental.dual_stream_graph.backend import compile_region
from torch_npu._experimental.dual_stream_graph.domain import (
    GraphExecutionDomain,
    current_domain,
)
from torch_npu.npu._graph_resource_pool import GraphResourcePool


class DualBatchRunner:
    """Run at most two argument groups on two isolated native graph trees.

    ``batch_independent=True`` is the caller's reviewed assertion that the
    region is inference-only, separable along the batch dimension, free of
    cross-batch reductions, shared writable state and RNG-dependent
    semantics, and does not need output aliases preserved.  This cannot be
    inferred from output shapes; parameters must stay read-only.

    ``lane_selector(groups)`` optionally maps each argument group to lane 0/1
    (e.g. to warm a small gear on its future lane).  It runs before Dynamo;
    the default assigns groups to lanes in order.

    Pass a raw function (or a plain ``def`` wrapping an ``nn.Module`` call),
    not a ``torch.compile`` wrapper; the runner compiles it under its own
    backend identity and no precompiled callable may be invoked inside.

    Call :meth:`run` outside ``torch.compile`` and under ``no_grad()`` /
    ``inference_mode()`` on the creating thread.  The caller stream must
    already depend on every input producer.  Adaptive graph resource eviction
    must stay disabled while the runner is used.  Close the runner when done.
    """

    def __init__(self, fn, *, batch_independent=False, lane_selector=None,
                 backend="inductor", dynamic=True, options=None):
        if batch_independent is not True:
            raise ValueError(
                "DualBatchRunner requires batch_independent=True for a reviewed "
                "inference region"
            )
        if lane_selector is not None and not callable(lane_selector):
            raise TypeError("lane_selector must be callable")
        self._fn, self._backend = compile_region(
            fn, backend=backend, options=options, dynamic=dynamic
        )
        self._lane_selector = lane_selector
        self._thread = threading.current_thread()
        self._domains = []
        self._streams = []
        self._device_index = None
        self._last_completion = None
        self._running = False
        self._closed = False

    # -- guards -------------------------------------------------------------

    def _check_idle(self):
        if threading.current_thread() is not self._thread:
            raise RuntimeError("DualBatchRunner must be used on its creating thread")
        if self._closed:
            raise RuntimeError("DualBatchRunner is closed")
        if self._running or current_domain() is not None:
            raise RuntimeError("Nested or reentrant DualBatchRunner calls are not supported")

    def _validate(self, groups, lanes):
        if not 1 <= len(groups) <= 2:
            raise ValueError("DualBatchRunner accepts one or two argument groups")
        if len(lanes) != len(groups) or any(
            type(lane) is not int or lane not in (0, 1) for lane in lanes
        ):
            raise ValueError("Each argument group must select lane 0 or 1")
        if len(set(lanes)) != len(lanes):
            raise ValueError("Concurrent argument groups must use different lanes")
        device_index = None
        input_tensors = []
        for args, kwargs in groups:
            if not isinstance(args, (tuple, list)) or not isinstance(kwargs, dict):
                raise TypeError("An argument group must be an (args, kwargs) pair")
            leaves, _ = tree_flatten((args, kwargs))
            tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
            if not tensors:
                raise ValueError("Each argument group must contain an NPU tensor")
            input_tensors.extend(tensors)
            for tensor in tensors:
                if tensor.device.type != "npu":
                    raise ValueError("DualBatchRunner only supports NPU inputs")
                if device_index is None:
                    device_index = tensor.device.index
                if tensor.device.index != device_index:
                    raise ValueError("DualBatchRunner cannot span devices")
        if self._device_index is not None and device_index != self._device_index:
            raise ValueError("Cannot change a DualBatchRunner's device")
        return device_index, input_tensors

    # -- output handling ----------------------------------------------------

    def _detach_output(self, output):
        """Clone tensor leaves on their lane; graph-owned memory stays inside."""

        def detach_leaf(leaf):
            if isinstance(leaf, torch.Tensor):
                if leaf.device.type != "npu" or leaf.device.index != self._device_index:
                    raise ValueError("DualBatchRunner requires outputs on its NPU device")
                return leaf.clone()
            if leaf is None or type(leaf) in (bool, int, float, str):
                return leaf
            raise TypeError(
                "DualBatchRunner only supports tensor pytrees and immutable scalars"
            )

        return tree_map(detach_leaf, output)

    # -- scheduling ---------------------------------------------------------

    @torch.compiler.disable(recursive=False)
    def run(self, groups, recover):
        self._check_idle()
        if torch.is_grad_enabled():
            raise RuntimeError("DualBatchRunner requires no_grad() or inference_mode()")
        groups = tuple(groups)
        if self._lane_selector is not None:
            lanes = tuple(self._lane_selector(groups))
        else:
            lanes = tuple(range(len(groups)))
        device_index, input_tensors = self._validate(groups, lanes)
        if GraphResourcePool.get_pool(device_index).is_active():
            raise RuntimeError(
                "DualBatchRunner does not support adaptive graph resource eviction"
            )
        if not callable(recover):
            raise TypeError("recover must be callable")
        self._running = True
        outputs = []
        try:
            with torch.npu.device(device_index):
                if not self._domains:
                    self._device_index = device_index
                    # Two lanes: two native containers -> two managers -> two
                    # private pools, created lazily on first capture.
                    self._domains = [GraphExecutionDomain(device_index) for _ in range(2)]
                    self._streams = [
                        torch.npu.Stream(device=device_index) for _ in range(2)
                    ]
                caller = torch.npu.current_stream()
                if self._last_completion is not None:
                    # Covers the previous recovery's reads even if the caller
                    # stream changed between requests.
                    caller.wait_event(self._last_completion)
                for (args, kwargs), lane in zip(groups, lanes):
                    stream = self._streams[lane]
                    stream.wait_stream(caller)
                    with torch.npu.stream(stream), self._domains[lane].activate():
                        outputs.append(self._detach_output(self._fn(*args, **kwargs)))
                for lane in lanes:
                    caller.wait_stream(self._streams[lane])
                # recover may consume/clear its argument; keep the cloned lane
                # tensors held here, including on its exception path.
                lane_output_leaves, _ = tree_flatten(outputs)
                result = recover(outputs)
                done = torch.npu.Event()
                done.record(caller)
                self._last_completion = done
                result_leaves, _ = tree_flatten(result)
                # Host return does not mean the copies/merge finished.  A
                # recover returning a lane clone (or view) keeps a private
                # allocation stream: register results on the caller stream so
                # reads submitted after run() returns stay ordered.
                for leaf in result_leaves:
                    if isinstance(leaf, torch.Tensor) and leaf.device.type == "npu":
                        leaf.record_stream(caller)
                self._domains[0].retain_until(
                    done, (input_tensors, lane_output_leaves, result_leaves)
                )
                return result
        except BaseException:
            # A lane may already have submitted device work.  Drain before the
            # references unwind and drop the partially built graph state.
            torch.npu.synchronize(device_index)
            for domain in self._domains:
                domain.reset()
            self._last_completion = None
            raise
        finally:
            self._running = False

    # -- lifecycle ----------------------------------------------------------

    def close(self):
        if self._closed:
            return
        self._check_idle()
        for domain in self._domains:
            domain.close()
        self._domains.clear()
        self._streams.clear()
        self._last_completion = None
        self._closed = True

    @property
    def domains(self):
        """The two lane domains (read-only observation, e.g. from tests)."""
        return tuple(self._domains)

    @property
    def compile_count(self):
        return self._backend.compile_count

    def __enter__(self):
        self._check_idle()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
