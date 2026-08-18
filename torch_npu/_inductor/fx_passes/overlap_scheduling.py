from __future__ import annotations

import functools
import inspect
import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.fx as fx


if TYPE_CHECKING:
    from collections.abc import Callable


_PATCHED_ATTR = "_torch_npu_overlap_scheduling_patched"
log = logging.getLogger(__name__)


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute the median of an empty list")
    return float(torch.median(torch.tensor(values)).item())


def _wait_collective_result(result: Any) -> None:
    torch.utils._pytree.tree_map_only(
        torch.Tensor,
        torch.ops._c10d_functional.wait_tensor,
        result,
    )


def _benchmark_callable_with_npu_events(
    fn: Callable[[], Any],
    *,
    warmup: int = 5,
    nruns: int = 3,
) -> float:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU runtime benchmarking requires an available NPU device")

    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    runtimes: list[float] = []
    for _ in range(nruns):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()
        runtimes.append(float(start_event.elapsed_time(end_event)))

    return _median(runtimes)


def _get_npu_do_bench() -> Callable[[Callable[[], Any]], float]:
    return functools.partial(
        _benchmark_callable_with_npu_events,
        warmup=5,
        nruns=3,
    )


def _benchmark_collective_with_npu_events_impl(
    node: fx.Node,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    nruns: int,
) -> float | None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return None

    torch.npu.synchronize()
    result = node.target(*args, **kwargs)  # type: ignore[operator]
    _wait_collective_result(result)
    torch.npu.synchronize()

    runtimes: list[float] = []
    for _ in range(nruns):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        result = node.target(*args, **kwargs)  # type: ignore[operator]
        _wait_collective_result(result)
        end_event.record()
        end_event.synchronize()
        runtimes.append(float(start_event.elapsed_time(end_event)))

    return _median(runtimes)


def _get_registered_npu_op_packet(name: str) -> object | None:
    try:
        packet = getattr(torch.ops.npu, name)
        _ = packet.default
        return packet
    except (AttributeError, RuntimeError):
        return None


def _build_npu_is_compute_node(
    upstream_is_compute_node: Callable[[fx.Node], bool],
) -> Callable[[fx.Node], bool]:
    npu_compute_packets = {
        _get_registered_npu_op_packet("npu_grouped_matmul"),
        _get_registered_npu_op_packet("npu_fusion_attention_v3"),
    }
    npu_compute_packets.discard(None)

    @functools.wraps(upstream_is_compute_node)
    def is_compute_node(node: fx.Node) -> bool:
        if upstream_is_compute_node(node):
            return True
        packet = getattr(node.target, "overloadpacket", node.target)
        return packet in npu_compute_packets

    return is_compute_node


def _unsupported_npu_roofline_estimation(node: fx.Node) -> float:
    # The upstream scheduler uses this value to seed its data structures even
    # when benchmark mode is selected. Benchmark results replace it before
    # scheduling, so zero is the safe placeholder for NPU.
    return 0.0


def _build_npu_estimate_collective_time(
    overlap_scheduling: Any,
    node_runtime_estimation: Any,
) -> Callable[..., float]:
    def estimate_collective_time(
        node: fx.Node,
        override_size: int | None = None,
        custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
        | None = None,
        collective_estimator: str = "analytical",
    ) -> float:
        custom_estimation = overlap_scheduling.get_custom_estimation(
            node,
            custom_runtime_estimation,
            override_size,
        )
        if custom_estimation is not None:
            return custom_estimation

        if collective_estimator == "benchmark":
            runtime, _ = node_runtime_estimation.benchmark_collective_with_cuda_events(
                node,
                nruns=5,
            )
            if runtime is not None:
                return runtime

        # OverlapScheduler seeds CollectiveInfo before cross-rank benchmark
        # alignment. NPU analytical estimation is intentionally unsupported;
        # the benchmark result replaces this placeholder before scheduling.
        return 0.0

    return estimate_collective_time


def _build_npu_collective_logger(
    runtime_estimation: Any,
) -> Callable[..., None]:
    def _log_collective_benchmarks(
        collective_nodes: list[fx.Node],
        collective_keys: list[str] | None = None,
        benchmarked_medians: list[float] | None = None,
        world_size: int | None = None,
        artifact_name: str = "fx_collectives_runtime_estimation",
    ) -> None:
        if world_size is None:
            world_size = (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else 1
            )

        headers = ["Collective Key", "Benchmarked(ms)"]
        rows: list[list[str]] = []
        for index, node in enumerate(collective_nodes):
            key = (
                collective_keys[index]
                if collective_keys is not None
                else runtime_estimation._get_collective_key(node)
            )
            benchmarked_ms = (
                benchmarked_medians[index] if benchmarked_medians is not None else 0.0
            )
            rows.append([key, f"{benchmarked_ms:.4f}"])

        log_text = f"# World size: {world_size}\n"
        log_text += runtime_estimation._format_csv(headers, rows)
        runtime_estimation.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": artifact_name,
                "encoding": "string",
            },
            payload_fn=lambda: log_text,
        )

    return _log_collective_benchmarks


def _build_npu_gather_node_runtime_estimations(
    upstream_gather: Callable[..., tuple[dict[fx.Node, float], dict[fx.Node, Any]]],
) -> Callable[..., tuple[dict[fx.Node, float], dict[fx.Node, Any]]]:
    signature = inspect.signature(upstream_gather)

    @functools.wraps(upstream_gather)
    def gather_node_runtime_estimations(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[dict[fx.Node, float], dict[fx.Node, Any]]:
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        enable_fusion_regions = bool(
            bound.arguments.get("enable_fusion_regions", False)
        )
        if not enable_fusion_regions:
            return upstream_gather(*args, **kwargs)

        # The upstream implementation builds CUDA/Inductor fusion regions and
        # mutates the FX graph. DVM predicts its own regions later in lowering,
        # so keep the ATen graph intact and only replace per-node cost estimates.
        bound.arguments["enable_fusion_regions"] = False
        estimations, region_of = upstream_gather(*bound.args, **bound.kwargs)

        from .dvm_fusion_regions import (
            build_dvm_fusion_regions,
            estimate_dvm_fused_node_costs,
        )

        dvm_region_of = build_dvm_fusion_regions(bound.arguments["gm"])
        fused_costs = estimate_dvm_fused_node_costs(dvm_region_of)
        estimations.update(fused_costs)
        log.info(
            "NPU DVM fusion region predictor: candidate_regions=%d, "
            "estimated_nodes=%d",
            len({id(region) for region in dvm_region_of.values()}),
            len(fused_costs),
        )
        return estimations, region_of

    return gather_node_runtime_estimations


def _build_npu_schedule_entry(
    upstream_schedule: Callable[..., fx.GraphModule],
) -> Callable[..., fx.GraphModule]:
    signature = inspect.signature(upstream_schedule)

    @functools.wraps(upstream_schedule)
    def schedule_overlap_bucketing(*args: Any, **kwargs: Any) -> fx.GraphModule:
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        compute_estimator = bound.arguments["compute_estimator"]
        collective_estimator = bound.arguments["collective_estimator"]

        if torch._inductor.config.deterministic:
            raise NotImplementedError(
                "NPU overlap scheduling does not support deterministic mode because "
                "it forces analytical runtime estimation"
            )
        if compute_estimator == "analytical":
            raise NotImplementedError(
                "NPU analytical compute runtime estimation is not implemented; "
                "use compute_estimator='benchmark' or a custom estimator"
            )
        if collective_estimator == "analytical":
            raise NotImplementedError(
                "NPU analytical collective runtime estimation is not implemented; "
                "use collective_estimator='benchmark' or a custom estimator"
            )

        return upstream_schedule(*args, **kwargs)

    return schedule_overlap_bucketing


def patch_overlap_scheduling() -> None:
    """Install the minimal NPU adapters for the PyTorch v2.12 upstream pass."""
    import torch._inductor.fx_passes.node_runtime_estimation as runtime_estimation
    import torch._inductor.fx_passes.overlap_scheduling as overlap_scheduling

    if getattr(overlap_scheduling, _PATCHED_ATTR, False):
        return

    upstream_is_compute_node = overlap_scheduling.is_compute_node
    upstream_gather = overlap_scheduling.gather_node_runtime_estimations
    upstream_schedule = overlap_scheduling.schedule_overlap_bucketing

    overlap_scheduling.get_collective_do_bench = _get_npu_do_bench
    runtime_estimation._benchmark_collective_with_cuda_events_impl = (
        _benchmark_collective_with_npu_events_impl
    )
    runtime_estimation._log_collective_benchmarks = _build_npu_collective_logger(
        runtime_estimation
    )
    overlap_scheduling.is_compute_node = _build_npu_is_compute_node(
        upstream_is_compute_node
    )
    overlap_scheduling.estimate_roofline_runtime_ms = (
        _unsupported_npu_roofline_estimation
    )
    overlap_scheduling.estimate_collective_time = _build_npu_estimate_collective_time(
        overlap_scheduling,
        runtime_estimation,
    )
    overlap_scheduling.gather_node_runtime_estimations = (
        _build_npu_gather_node_runtime_estimations(upstream_gather)
    )
    overlap_scheduling.schedule_overlap_bucketing = _build_npu_schedule_entry(
        upstream_schedule
    )

    # Use the supported NPU estimators by default. All other v2.12 options are
    # left untouched and continue to be handled by the upstream pass.
    dist_opts = torch._inductor.config.aten_distributed_optimizations
    dist_opts.compute_estimator = "benchmark"
    dist_opts.collective_estimator = "benchmark"
    dist_opts.enable_fusion_regions = False

    setattr(overlap_scheduling, _PATCHED_ATTR, True)
