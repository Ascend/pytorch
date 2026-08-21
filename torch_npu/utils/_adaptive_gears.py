from __future__ import annotations

import copy
import logging
import math
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch_npu.npu._graph_resource_pool import GraphResourcePool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class GearEventSample:
    """Single request event sample produced by shape handling, used as input for
    time-decayed window statistics (Section 5).

    Fields:
        event_ts: Event timestamp in seconds, used for time-decay weight w_t(e) = exp(-λ · Δt).
        raw_value: Original dimension value of the request, used for raw length distribution
            to generate addition candidates via median.
        pad_ratio: Padding ratio of this request, used for gear weighted average pad cost.
        split_ratio: Split ratio of this request, used for gear weighted average split cost.
    """

    event_ts: float
    raw_value: int
    pad_ratio: float
    split_ratio: float


@dataclass
class GearRuntimeState:
    """Per-gear runtime state storing decision data for eviction evaluation ("whether to evict"),
    not deletion safety ("whether safe to delete") which is handled by tree manager.

    Fields:
        gear_id: Unique gear identifier in the format "{shape_type}:{gear_value}".
        shape_type: Dimension type this gear belongs to (e.g. "batch_size", "seq_len").
        gear_value: The gear threshold value.
        samples: Windowed event sample list, supporting hit_rate, avg_pad, avg_split computation.
        created_ts: Gear creation timestamp, used for recent-use protection window.
        last_hit_ts: Timestamp of the most recent hit, used for recent-use protection window.
    """

    gear_id: str
    shape_type: str
    gear_value: int
    samples: List[GearEventSample] = field(default_factory=list)
    cleanup_keys: Set[int] = field(default_factory=set)
    created_ts: float = 0.0
    last_hit_ts: float = 0.0


@dataclass
class GearSnapshot:
    """Snapshot of the current gear set, enabling lock-free reads from the request thread
    via clone-on-read and atomic reference swaps on publish.

    Fields:
        active_gears: Current active gears, keyed by shape_type, values are gear value lists.
        handler_configs: Per-dimension shape handler configs, used to rebuild config on update commit.
        shape_handling: Associated NPUShapeHandling instance for accessing graph manager and runtime config.
        version: Monotonically increasing version number, assigned at snapshot creation time.
            Useful for correlating logs across request and update threads.
        created_at: Unix timestamp (``time.time()``) when the snapshot was built.
    """

    active_gears: Dict[str, List[int]]
    handler_configs: List[Dict[str, Any]]
    shape_handling: Any
    version: int = 0
    created_at: float = 0.0

    def clone(self) -> "GearSnapshot":
        """Deep-copy the snapshot. active_gears and handler_configs are independent copies;
        shape_handling shares the reference.  version and created_at are preserved."""
        return GearSnapshot(
            active_gears={shape_type: list(values) for shape_type, values in self.active_gears.items()},
            handler_configs=copy.deepcopy(self.handler_configs),
            shape_handling=self.shape_handling,
            version=self.version,
            created_at=self.created_at,
        )

@dataclass
class ScoreBreakdown:
    """Scoring result for a single gear, used for eviction ranking, replacement loss recording,
    and observability logging (Section 12.2).

    Fields:
        gear_id: Unique gear identifier.
        shape_type: Dimension type this gear belongs to.
        gear_value: The gear threshold value.
        hit_rate: Hit rate score component.
        avg_pad_ratio: Weighted average padding ratio component.
        avg_split_ratio: Weighted average split ratio component.
        score: Composite score = w_h * hit_rate - w_p * avg_pad - w_sp * avg_split.
        replace_loss: Replacement loss. Weighted average loss incurred when this gear's samples
            are remapped to the nearest remaining gear after deletion.
        replacement_gear_id: The best alternative gear_id identified during replacement loss computation.
    """

    gear_id: str
    shape_type: str
    gear_value: int
    hit_rate: float
    avg_pad_ratio: float
    avg_split_ratio: float
    score: float
    replace_loss: float = 0.0
    replacement_gear_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class GearSnapshotStore:
    """Thread-safe store for the current gear snapshot.

    Reads are lock-free: a snapshot reference is read atomically (GIL) and
    cloned on-read, so the request thread never blocks on writes.  Writes
    are serialized by a lock to prevent concurrent publishes from interleaving.
    """

    def __init__(self, initial_state: GearSnapshot) -> None:
        self._lock = threading.Lock()
        self._current_snapshot = initial_state.clone()

    def get_snapshot(self) -> GearSnapshot:
        return self._current_snapshot.clone()

    def publish_snapshot(self, snapshot: GearSnapshot) -> GearSnapshot:
        with self._lock:
            self._current_snapshot = snapshot.clone()
            return self._current_snapshot


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class GearScorer:
    """Scores gears based on time-decayed hit rate, padding cost, and split cost.

    Produces a ``ScoreBreakdown`` per gear used for eviction ranking, and computes
    replacement loss to decide whether a specific gear is safe to remove.
    """

    def __init__(self, adaptive_configs: Dict[str, float]) -> None:
        self.weight_hit = adaptive_configs.get("weight_hit", 0.60)
        self.weight_pad = adaptive_configs.get("weight_pad", 0.20)
        self.weight_split = adaptive_configs.get("weight_split", 0.20)

    def build_score_breakdown(
        self,
        snapshot: GearSnapshot,
        stats_snapshot: Dict[str, Dict],
    ) -> Dict[str, ScoreBreakdown]:
        totals_by_type: Dict[str, float] = {}
        for stat in stats_snapshot.values():
            totals_by_type.setdefault(stat["shape_type"], 0.0)
            totals_by_type[stat["shape_type"]] += stat["weighted_hits"]

        breakdowns: Dict[str, ScoreBreakdown] = {}
        for shape_type, gear_values in snapshot.active_gears.items():
            total_hits = max(totals_by_type.get(shape_type, 0.0), 1e-12)
            for gear_value in gear_values:
                gear_id = f"{shape_type}:{gear_value}"
                stat = stats_snapshot.get(gear_id)
                if stat is None:
                    hit_rate = 0.0
                    avg_pad_ratio = 0.0
                    avg_split_ratio = 0.0
                else:
                    hit_rate = stat["weighted_hits"] / total_hits
                    avg_pad_ratio = stat["avg_pad_ratio"]
                    avg_split_ratio = stat["avg_split_ratio"]
                score = (
                    self.weight_hit * hit_rate
                    - self.weight_pad * avg_pad_ratio
                    - self.weight_split * avg_split_ratio
                )
                breakdowns[gear_id] = ScoreBreakdown(
                    gear_id=gear_id,
                    shape_type=shape_type,
                    gear_value=gear_value,
                    hit_rate=hit_rate,
                    avg_pad_ratio=avg_pad_ratio,
                    avg_split_ratio=avg_split_ratio,
                    score=score,
                )
        return breakdowns

    def compute_replace_loss(
        self,
        snapshot: GearSnapshot,
        stats_snapshot: Dict[str, Dict],
        gear_id: str,
    ) -> Tuple[float, Optional[str]]:
        stat = stats_snapshot.get(gear_id)
        if stat is None:
            return 0.0, None

        shape_type = stat["shape_type"]
        gear_value = stat["gear_value"]
        alternatives = [
            candidate
            for candidate in snapshot.active_gears.get(shape_type, [])
            if candidate != gear_value
        ]
        if not alternatives:
            return float("inf"), None

        raw_samples = stat.get("raw_samples", [])
        if not raw_samples:
            replacement = min(alternatives, key=lambda candidate: abs(candidate - gear_value))
            return 0.0, f"{shape_type}:{replacement}"

        best_loss = float("inf")
        best_replacement = None
        for candidate in alternatives:
            total_loss = 0.0
            for raw_value in raw_samples:
                if candidate >= raw_value:
                    total_loss += (candidate - raw_value) / max(candidate, 1)
                else:
                    total_loss += (raw_value - candidate) / max(raw_value, 1)
            avg_loss = total_loss / max(len(raw_samples), 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_replacement = candidate

        if best_replacement is None:
            return float("inf"), None
        return best_loss, f"{shape_type}:{best_replacement}"


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class AdaptiveGearRuntime:
    """Central coordinator for the adaptive gear update mechanism.

    Manages the main request-path logic (event recording, snapshot reads, update
    triggering) and owns the worker thread pool, scorer, and snapshot store.
    Graph cleanup after gear eviction is delegated to
    GraphResourcePool.remove_by_keys.
    """

    DEFAULT_CONFIG = {
        "window_seconds": 300.0,
        "update_interval_seconds": 60.0,
        "pad_add_threshold": 0.35,
        "split_add_threshold": 0.20,
        "min_samples_per_gear": 5,
        "add_min_samples": 5,
        "min_gear_count_per_type": 1,
        "max_gears_per_type": 64,
        "recent_use_protect_seconds": 300.0,
        "replace_loss_threshold": 0.60,
        "weight_hit": 0.60,
        "weight_pad": 0.20,
        "weight_split": 0.20,
        "device_memory_usage_threshold_ratio": 0.90,
    }

    def __init__(
        self,
        shape_handling_configs: List[Dict[str, Any]],
        adaptive_configs: Dict[str, Any],
        shape_handling_builder: Callable[[List[Dict[str, Any]]], Any],
        snapshot_store: Optional[GearSnapshotStore] = None,
        scorer: Optional[GearScorer] = None,
    ) -> None:
        self.device_index = torch.npu.current_device()
        self.shape_handling_builder = shape_handling_builder
        self.config = dict(self.DEFAULT_CONFIG)
        if adaptive_configs:
            self.config.update(adaptive_configs)

        self.scorer = scorer or GearScorer(self.config)
        self._sample_lock = threading.Lock()
        self._commit_lock = threading.Lock()
        self._snapshot_version = 0

        normalized_configs = self._normalize_configs(shape_handling_configs)
        self.config_by_type = {config["type"]: config for config in normalized_configs}
        self._cached_ordered_configs = [
            self.config_by_type[shape_type] for shape_type in sorted(self.config_by_type.keys())
        ]
        initial_snapshot = self._build_snapshot(normalized_configs)
        self.snapshot_store = snapshot_store or GearSnapshotStore(initial_snapshot)
        self._states: Dict[str, GearRuntimeState] = {}
        self._ensure_runtime_states(initial_snapshot)

        self.worker = GearUpdateWorker(self)

        self._shutdown_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="adaptive-gears-loop",
        )
        self._worker_thread.start()

    def shutdown(self) -> None:
        self._shutdown_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

    def get_snapshot(self) -> GearSnapshot:
        return self.snapshot_store.get_snapshot()

    def record_event(
        self,
        raw_gear_values: List[List[Optional[int]]],
        mapped_gear_values: List[List[Optional[int]]],
        pad_ratios: List[List[float]],
        split_ratios: List[List[float]],
        event_ts: float,
        cleanup_key: Optional[int] = None,
    ) -> None:
        if not raw_gear_values or not mapped_gear_values:
            return
        with self._sample_lock:
            for index, config in enumerate(self._ordered_configs()):
                raw_per_tensor = self._value_at(raw_gear_values, index, default=[])
                mapped_per_tensor = self._value_at(mapped_gear_values, index, default=[])
                cfg_pad = self._value_at(pad_ratios, index, default=[])
                cfg_split = self._value_at(split_ratios, index, default=[])
                for tensor_idx, (raw_value, mapped_value) in enumerate(
                    zip(raw_per_tensor, mapped_per_tensor)
                ):
                    if raw_value is None or mapped_value is None:
                        continue
                    gear_id = f"{config['type']}:{mapped_value}"
                    state = self._states.get(gear_id)
                    if state is None:
                        state = GearRuntimeState(
                            gear_id=gear_id,
                            shape_type=config["type"],
                            gear_value=mapped_value,
                        )
                        self._states[gear_id] = state
                    sample = GearEventSample(
                        event_ts=event_ts,
                        raw_value=raw_value,
                        pad_ratio=float(self._value_at(cfg_pad, tensor_idx, default=0.0) or 0.0),
                        split_ratio=float(self._value_at(cfg_split, tensor_idx, default=0.0) or 0.0),
                    )
                    state.samples.append(sample)
                    if cleanup_key is not None:
                        state.cleanup_keys.add(cleanup_key)
                    state.last_hit_ts = event_ts

    def build_stats_snapshot(self, now_ts: float) -> Dict[str, Dict[str, Any]]:
        decay_lambda = math.log(2.0) / max(self.config["window_seconds"] / 2.0, 1e-6)
        window_seconds = self.config["window_seconds"]
        stats_snapshot: Dict[str, Dict[str, Any]] = {}
        with self._sample_lock:
            for gear_id, state in self._states.items():
                weighted_hits = 0.0
                weighted_pad_sum = 0.0
                weighted_split_sum = 0.0
                raw_samples = []
                kept_samples = []
                pad_sample_count = 0
                split_sample_count = 0
                for sample in state.samples:
                    age = now_ts - sample.event_ts
                    if age > window_seconds:
                        continue
                    weight = math.exp(-decay_lambda * max(age, 0.0))
                    weighted_hits += weight
                    weighted_pad_sum += sample.pad_ratio * weight
                    weighted_split_sum += sample.split_ratio * weight
                    raw_samples.append(sample.raw_value)
                    if sample.pad_ratio > 0.0:
                        pad_sample_count += 1
                    if sample.split_ratio > 0.0:
                        split_sample_count += 1
                    kept_samples.append(sample)
                state.samples = kept_samples
                avg_pad_ratio = weighted_pad_sum / max(weighted_hits, 1e-12)
                avg_split_ratio = weighted_split_sum / max(weighted_hits, 1e-12)
                stats_snapshot[gear_id] = {
                    "gear_id": gear_id,
                    "shape_type": state.shape_type,
                    "gear_value": state.gear_value,
                    "weighted_hits": weighted_hits,
                    "avg_pad_ratio": avg_pad_ratio,
                    "avg_split_ratio": avg_split_ratio,
                    "sample_count": len(raw_samples),
                    "pad_sample_count": pad_sample_count,
                    "split_sample_count": split_sample_count,
                    "raw_samples": raw_samples,
                    "created_ts": state.created_ts,
                    "last_hit_ts": state.last_hit_ts,
                }
        current_snapshot = self.get_snapshot()
        for shape_type, values in current_snapshot.active_gears.items():
            for gear_value in values:
                gear_id = f"{shape_type}:{gear_value}"
                stats_snapshot.setdefault(
                    gear_id,
                    {
                        "gear_id": gear_id,
                        "shape_type": shape_type,
                        "gear_value": gear_value,
                        "weighted_hits": 0.0,
                        "avg_pad_ratio": 0.0,
                        "avg_split_ratio": 0.0,
                        "sample_count": 0,
                        "pad_sample_count": 0,
                        "split_sample_count": 0,
                        "raw_samples": [],
                        "created_ts": 0.0,
                        "last_hit_ts": 0.0,
                    },
                )
        return stats_snapshot

    def _update_loop(self) -> None:
        check_interval = self.config["update_interval_seconds"]
        while not self._shutdown_event.wait(timeout=check_interval):
            try:
                self.worker.run_once(time.time())
            except Exception:
                logger.warning("Adaptive gear update failed", exc_info=True)

    def protect_gear_from_eviction(self, gear_id: str, now_ts: float) -> None:
        with self._sample_lock:
            state = self._states.get(gear_id)
            if state is not None:
                state.last_hit_ts = max(state.last_hit_ts, now_ts)

    def commit_update(
        self,
        configs: List[Dict[str, Any]],
        removed_gears: List[str],
        now_ts: float = 0.0,
    ) -> Tuple[Optional[GearSnapshot], List[int]]:
        next_snapshot = self._build_snapshot(configs)
        published_snapshot = self.snapshot_store.publish_snapshot(next_snapshot)
        self._ensure_runtime_states(published_snapshot, created_ts=now_ts)
        removed_keys = self._collect_cleanup_keys(removed_gears)
        with self._sample_lock:
            for gear_id in removed_gears:
                self._states.pop(gear_id, None)
        return published_snapshot, removed_keys

    def build_resource_budget(self) -> Dict[str, Any]:
        device_memory_threshold = self._normalize_ratio(self.config.get("device_memory_usage_threshold_ratio"))
        device_memory_usage_ratio = self._get_device_memory_usage_ratio()
        return {
            "device_memory_usage_ratio": device_memory_usage_ratio,
            "device_memory_usage_threshold_ratio": device_memory_threshold,
            "device_memory_usage_high": (
                device_memory_threshold is not None
                and device_memory_usage_ratio is not None
                and device_memory_usage_ratio >= device_memory_threshold
            ),
        }

    def _ordered_configs(self) -> List[Dict[str, Any]]:
        return self._cached_ordered_configs

    def _normalize_configs(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize user-supplied shape-handling configs into a canonical form.

        Two paths:
        1. *Explicit gears* — the user provides a ``gears`` list.  Values are
           deduplicated, cast to int, and sorted.
        2. *TIMES policy* — ``gears`` is empty; sizes are auto-generated from
           ``min_size``, ``max_size`` and the TIMES (power-of-two) strategy via
           ``_expand_policy_gears``.  The expansion algorithm is kept consistent
           with ``NPUShapeHandling::GenerateGears`` in the C++ layer so that the
           initial gear set derived here matches what the C++ engine would
           produce internally.

        The function modifies the config dicts in place and returns the same
        list (or ``[]`` for empty / None input).
        """
        if not configs:
            return []
        for config in configs:
            gears = config.get("gears") or self._expand_policy_gears(config)
            config["gears"] = sorted(set(int(gear) for gear in gears))
        return configs

    def _expand_policy_gears(self, config: Dict[str, Any]) -> List[int]:
        """Expand ``min_size`` .. ``max_size`` using the TIMES (power-of-two) strategy.

        The factor 2 is chosen because it keeps the gear count logarithmic
        relative to the size range — each gear is at most 2× the previous one,
        bounding the worst-case padding waste to ≤ 50 % while avoiding an
        explosion of graph variants.  For example, range [1, 64] produces only
        7 gears: [1, 2, 4, 8, 16, 32, 64].

        This implementation mirrors ``NPUShapeHandling::GenerateGears`` in C++:
        it anchors at ``min_size``, then jumps to the next power of two and
        doubles from there, finally appending ``max_size`` if it was not already
        covered.
        """
        import math

        min_size = int(config.get("min_size", 1))
        max_size = int(config.get("max_size", min_size))
        if max_size <= min_size:
            return [min_size]
        gears = [min_size]
        exp = math.ceil(math.log2(min_size)) if min_size > 0 else 0
        gear = 1 << exp
        if gear == min_size:
            gear <<= 1
        while gear > 0 and gear <= max_size:
            gears.append(gear)
            if gear > max_size // 2:
                break
            gear <<= 1
        if gears[-1] != max_size:
            gears.append(max_size)
        return gears

    def _build_snapshot(self, configs: List[Dict[str, Any]]) -> GearSnapshot:
        self._snapshot_version += 1
        active_gears = {config["type"]: list(config.get("gears", [])) for config in configs}
        shape_handling = self.shape_handling_builder(copy.deepcopy(configs))
        return GearSnapshot(
            active_gears=active_gears,
            handler_configs=copy.deepcopy(configs),
            shape_handling=shape_handling,
            version=self._snapshot_version,
            created_at=time.time(),
        )

    def _ensure_runtime_states(self, snapshot: GearSnapshot, created_ts: float = 0.0) -> None:
        with self._sample_lock:
            for shape_type, gear_values in snapshot.active_gears.items():
                for gear_value in gear_values:
                    gear_id = f"{shape_type}:{gear_value}"
                    if gear_id not in self._states:
                        self._states[gear_id] = GearRuntimeState(
                            gear_id=gear_id,
                            shape_type=shape_type,
                            gear_value=gear_value,
                            created_ts=created_ts,
                        )

    def _collect_cleanup_keys(self, gear_ids: List[str]) -> List[int]:
        keys: set[int] = set()
        with self._sample_lock:
            for gear_id in gear_ids:
                state = self._states.get(gear_id)
                if state is None:
                    continue
                keys.update(state.cleanup_keys)
                state.cleanup_keys.clear()
        return list(keys)

    def _normalize_ratio(self, ratio_value):
        if ratio_value is None:
            return None
        try:
            normalized = float(ratio_value)
        except (TypeError, ValueError):
            return None
        if normalized <= 0.0 or normalized > 1.0:
            return None
        return normalized

    def _get_device_memory_usage_ratio(self):
        try:
            free_bytes, total_bytes = torch.npu.mem_get_info()
        except Exception:
            logger.debug("Failed to get device memory info", exc_info=True)
            return None
        if total_bytes <= 0:
            return None
        usage_ratio = 1.0 - (float(free_bytes) / float(total_bytes))
        return min(max(usage_ratio, 0.0), 1.0)

    def _value_at(self, values, index: int, default=None):
        if isinstance(values, (list, tuple)) and index < len(values):
            return values[index]
        return default


# ---------------------------------------------------------------------------
# Update Worker
# ---------------------------------------------------------------------------


class GearUpdateWorker:
    """Asynchronous worker that executes gear update decisions.

    Runs inside the worker thread pool. ``run_once`` performs scoring, eviction
    candidate selection, addition candidate generation, and commit -- all under
    the commit lock. After the lock is released, calls
    GraphResourcePool.remove_by_shapes to safely clean up evicted gear paths.

    Parameters:
        runtime: The owning AdaptiveGearRuntime instance. Provides access to the
            snapshot store, sample lock, config, scorer, and device_index.
    """

    def __init__(self, runtime: AdaptiveGearRuntime) -> None:
        self.runtime = runtime
        self.config = runtime.config
        self.scorer = runtime.scorer

    def run_once(self, now_ts: float):
        """Execute one cycle of the adaptive gear update algorithm.

        Called periodically by the daemon thread at an interval controlled by
        ``update_interval_seconds`` (default: 60.0 s).  Each cycle:

        1. Acquires the commit lock and snapshots the current state.
        2. Computes time-decayed stats, scores each gear, and selects eviction
           and addition candidates.
        3. Applies the update via ``commit_update``, which builds a new snapshot
           and publishes it atomically.
        4. Releases the lock, then cleans up any evicted graph resources via
           ``GraphResourcePool.remove_by_shapes``.

        The call is a no-op when there are no candidate changes to apply.
        """
        removed_keys: List[int] = []
        with self.runtime._commit_lock:
            snapshot = self.runtime.get_snapshot()
            stats_snapshot = self.runtime.build_stats_snapshot(now_ts)
            score_breakdowns = self.scorer.build_score_breakdown(snapshot, stats_snapshot)
            candidate_evictions = self.build_eviction_candidates(score_breakdowns, snapshot, stats_snapshot, now_ts)
            candidate_additions = self.build_addition_candidates(snapshot, stats_snapshot)
            resource_budget = self.runtime.build_resource_budget()
            result = self.commit_update(
                snapshot, stats_snapshot,
                candidate_evictions, candidate_additions, resource_budget, now_ts,
            )
            if result is not None:
                _, removed_keys = result

        if removed_keys:
            self._cleanup_graphs(removed_keys, self.runtime.device_index)
        return result

    @staticmethod
    def _cleanup_graphs(keys: List[int], device_index: int) -> None:
        """Release graph resources associated with evicted gears.

        Pool keys (``(model_sig, tensor_sigs)`` tuples) are looked up in the
        per-device ``GraphResourcePool`` and removed.  For each matched
        resource the pool calls ``release()`` (tree mode — ``NPUGraphNode``,
        which resets the underlying NPU graph and recycles block pointers) or
        ``reset()`` (simple mode — ``torch.npu.NPUGraph``).  The
        ``torch.npu.synchronize()`` call before each release ensures all
        outstanding device work on those graphs has completed, preventing
        use-after-free.

        Parameters:
            keys: Pool-key tuples previously registered via
                ``GraphResourcePool.register``.
            device_index: NPU device whose pool should be cleaned up.  Must
                match the device on which the graphs were originally recorded.
        """
        GraphResourcePool.get_pool(device_index).remove_by_keys(keys)

    def build_eviction_candidates(
        self,
        breakdowns: Dict[str, ScoreBreakdown],
        snapshot: GearSnapshot,
        stats_snapshot: Dict[str, Dict],
        now_ts: float,
    ) -> Dict[str, str]:
        max_gear_by_type = {
            shape_type: max(values)
            for shape_type, values in snapshot.active_gears.items()
            if values
        }
        candidates_by_type: Dict[str, List[Tuple[float, str]]] = {}
        for gear_id, breakdown in breakdowns.items():
            if breakdown.gear_value == max_gear_by_type.get(breakdown.shape_type):
                continue
            stat = stats_snapshot.get(
                gear_id,
                {
                    "created_ts": 0.0,
                    "last_hit_ts": 0.0,
                },
            )
            protect_anchor_ts = max(stat.get("created_ts", 0.0), stat.get("last_hit_ts", 0.0))
            if now_ts - protect_anchor_ts < self.config["recent_use_protect_seconds"]:
                continue
            if len(snapshot.active_gears.get(breakdown.shape_type, [])) <= self.config["min_gear_count_per_type"]:
                continue
            candidates_by_type.setdefault(breakdown.shape_type, []).append((breakdown.score, gear_id))
        selected_candidates = {}
        for shape_type, candidates in candidates_by_type.items():
            candidates.sort(key=lambda item: item[0])
            selected_candidates[shape_type] = candidates[0][1]
        return selected_candidates

    def build_addition_candidates(
        self,
        snapshot: GearSnapshot,
        stats_snapshot: Dict[str, Dict],
    ) -> List[Tuple[float, int, str, int]]:
        max_gear_by_type = {
            shape_type: max(values)
            for shape_type, values in snapshot.active_gears.items()
            if values
        }
        candidates: List[Tuple[float, int, str, int]] = []

        for stat in stats_snapshot.values():
            shape_type = stat["shape_type"]
            gear_value = stat["gear_value"]
            active_gears = set(snapshot.active_gears.get(shape_type, []))
            raw_samples = stat.get("raw_samples", [])
            if not raw_samples:
                continue
            is_max_gear = gear_value == max_gear_by_type.get(shape_type)

            if is_max_gear:
                enough_split = stat.get("split_sample_count", 0) >= self.config["add_min_samples"]
                high_split = stat["avg_split_ratio"] >= self.config["split_add_threshold"]
                if enough_split and high_split:
                    g_new = int(statistics.median(raw_samples))
                    if g_new not in active_gears:
                        pressure = stat["avg_split_ratio"] - self.config["split_add_threshold"]
                        candidates.append((pressure, stat["sample_count"], shape_type, g_new))

            enough_pad = stat.get("pad_sample_count", 0) >= self.config["add_min_samples"]
            high_pad = stat["avg_pad_ratio"] >= self.config["pad_add_threshold"]
            if enough_pad and high_pad:
                g_new = int(statistics.median(raw_samples))
                if g_new not in active_gears:
                    pressure = stat["avg_pad_ratio"] - self.config["pad_add_threshold"]
                    candidates.append((pressure, stat["sample_count"], shape_type, g_new))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates

    def commit_update(
        self,
        snapshot: GearSnapshot,
        stats_snapshot: Dict[str, Dict],
        candidate_evictions: Dict[str, str],
        candidate_additions: List[Tuple[float, int, str, int]],
        resource_budget: Dict[str, Any],
        now_ts: float,
    ) -> Optional[Tuple[Optional[GearSnapshot], List[Tuple[Tuple[int, ...], ...]]]]:
        resource_pressure_high = bool(resource_budget.get("device_memory_usage_high", False))
        next_configs = copy.deepcopy(snapshot.handler_configs)
        next_active_gears = {
            shape_type: set(values)
            for shape_type, values in snapshot.active_gears.items()
        }
        removed_gears: List[str] = []
        added = False

        for shape_type, gear_id in candidate_evictions.items():
            gear_value = int(gear_id.split(":", 1)[1])
            replace_loss, _ = self.scorer.compute_replace_loss(
                snapshot,
                stats_snapshot,
                gear_id,
            )
            if replace_loss > self.config["replace_loss_threshold"]:
                self.runtime.protect_gear_from_eviction(gear_id, now_ts)
                continue
            next_active_gears[shape_type].discard(gear_value)
            removed_gears.append(gear_id)

        addition_budget = None if not resource_pressure_high else len(removed_gears)
        for _, _, shape_type, candidate in candidate_additions:
            if addition_budget is not None and addition_budget <= 0:
                break
            if candidate in next_active_gears.setdefault(shape_type, set()):
                continue
            # Enforce per-type gear count cap to prevent unbounded graph variant growth
            # (each gear produces a separate compiled graph, consuming device memory).
            if len(next_active_gears.get(shape_type, set())) >= self.config["max_gears_per_type"]:
                continue
            next_active_gears[shape_type].add(candidate)
            added = True
            if addition_budget is not None:
                addition_budget -= 1

        if not removed_gears and not added:
            return None

        for config in next_configs:
            active_values = sorted(next_active_gears.get(config["type"], set()))
            config["gears"] = active_values
            config["policy"] = "CUSTOM"

        return self.runtime.commit_update(next_configs, removed_gears, now_ts=now_ts)


# ---------------------------------------------------------------------------
# Metadata collection utilities
# ---------------------------------------------------------------------------


def _resolve_dimension(config: Dict[str, Any], dimensions: List[int], position: int) -> int:
    if config.get("type") == "BATCHSIZE":
        if len(dimensions) == 0:
            return 0
        return dimensions[0]
    if len(dimensions) == 0:
        return 1
    if position < len(dimensions):
        return dimensions[position]
    return dimensions[0]


def _extract_gear_shapes(tensors: List[torch.Tensor], configs: List[Dict[str, Any]]) -> List[List[Optional[int]]]:
    all_shapes: List[List[Optional[int]]] = []
    for config in configs:
        indices = config.get("indices", [])
        dimensions = config.get("dimensions", [])
        target_indices = indices if len(indices) > 0 else list(range(len(tensors)))
        config_values: List[Optional[int]] = []
        for position, tensor_index in enumerate(target_indices):
            if tensor_index >= len(tensors):
                config_values.append(None)
                continue
            tensor = tensors[tensor_index]
            dimension = _resolve_dimension(config, dimensions, position)
            if tensor.ndim > dimension:
                config_values.append(tensor.shape[dimension])
            else:
                config_values.append(None)
        all_shapes.append(config_values)
    return all_shapes


def collect_transform_metadata(
    inputs: List[torch.Tensor],
    trans_outputs: List[List[torch.Tensor]],
    configs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collect per-variant pad / split metadata for adaptive gear recording.

    The *cleanup_key* is no longer constructed here — it is obtained from
    ``GraphResourcePool.consume_recent_keys()`` after ``src_fn`` returns.
    """
    # Sort configs by type to match AdaptiveGearRuntime._ordered_configs().
    sorted_configs = sorted(configs, key=lambda c: c["type"])
    raw_gear_values = _extract_gear_shapes(inputs, sorted_configs)
    metadata = []
    for output_group in trans_outputs:
        mapped_gear_values = _extract_gear_shapes(output_group, sorted_configs)
        pad_ratios: List[List[float]] = []
        split_ratios: List[List[float]] = []
        for raw_per_cfg, mapped_per_cfg in zip(raw_gear_values, mapped_gear_values):
            cfg_pad: List[float] = []
            cfg_split: List[float] = []
            for raw_value, mapped_value in zip(raw_per_cfg, mapped_per_cfg):
                if mapped_value is None:
                    cfg_pad.append(0.0)
                    cfg_split.append(0.0)
                    continue
                r = raw_value if raw_value is not None else mapped_value
                cfg_pad.append(max(mapped_value - r, 0) / max(mapped_value, 1))
                cfg_split.append(max(r - mapped_value, 0) / max(r, 1))
            pad_ratios.append(cfg_pad)
            split_ratios.append(cfg_split)

        metadata.append(
            {
                "raw_gear_values": raw_gear_values,
                "mapped_gear_values": mapped_gear_values,
                "pad_ratios": pad_ratios,
                "split_ratios": split_ratios,
            }
        )
    return metadata
