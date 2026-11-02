import csv
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
import glob
from typing import Callable, List, Optional, Tuple

import torch
import triton

from torch_npu.profiler.analysis.prof_common_func._constant import print_warn_msg, print_info_msg
from torch_npu.profiler.analysis.prof_common_func._utils import no_exception_func
from torch_npu.profiler._profiler_path_creator import ProfPathCreator

try:
    from mspti import KernelMonitor
except ImportError:
    KernelMonitor = None


log = logging.getLogger("torch._inductor")
_NPU_PROFILER_BENCHMARK_LOCK = threading.Lock()
_NPU_PROFILER_BENCHMARK_STATE = threading.local()
_CV_BENCHMARK_BACKEND_ENV = "TORCHINDUCTOR_NPU_CV_BENCHMARK_BACKEND"
_CV_BENCHMARK_STRATEGY = "fallback"
_CV_BENCHMARK_FALLBACK_BACKENDS = ("mspti", "batch_profile", "benchmarker")


def get_npu_cv_benchmark_backend() -> str:
    """Return the timing strategy used only by CV fusion profitability checks."""
    configured = os.getenv(_CV_BENCHMARK_BACKEND_ENV, _CV_BENCHMARK_STRATEGY)
    configured = configured.strip().lower()
    if configured not in (_CV_BENCHMARK_STRATEGY, "auto"):
        raise ValueError(
            f"Unsupported {_CV_BENCHMARK_BACKEND_ENV}={configured!r}; CV "
            "benchmarking now uses the fixed fallback strategy "
            "mspti -> batch_profile -> benchmarker."
        )
    return _CV_BENCHMARK_STRATEGY


def get_npu_cv_benchmark_cache_tag() -> str:
    """Return a stable cache tag so fallback timings never mix with old clocks."""
    mspti_state = (
        "mspti_preloaded"
        if "libmspti.so" in os.getenv("LD_PRELOAD", "")
        else "mspti_not_preloaded"
    )
    return f"npu_cv_fallback_mspti_batch_profile_benchmarker_{mspti_state}_v1"


def npu_profiler_session_active():
    """Return whether this process already owns an NPU profiler session."""
    try:
        return bool(ProfPathCreator().is_prof_inited)
    except Exception:
        return False


def _read_kernel_duration_sum_ms(profile_root: str) -> float:
    """Read one profiler sample and return total NPU kernel time in ms."""
    kernel_detail_files = []
    for root, _, files in os.walk(profile_root):
        for file_name in files:
            if file_name == "kernel_details.csv":
                kernel_detail_files.append(os.path.join(root, file_name))

    if len(kernel_detail_files) != 1:
        raise RuntimeError(
            "Expected exactly one kernel_details.csv under "
            f"{profile_root}, found {len(kernel_detail_files)}."
        )

    total_us = 0.0
    row_count = 0
    with open(kernel_detail_files[0], newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"Name", "Duration(us)"}
        if not required_columns.issubset(reader.fieldnames or ()):
            raise RuntimeError(
                f"Profiler result {kernel_detail_files[0]} is missing columns "
                f"{sorted(required_columns)}."
            )
        for row in reader:
            duration_text = (row.get("Duration(us)") or "").strip()
            if not duration_text:
                continue
            duration_us = float(duration_text)
            if not math.isfinite(duration_us) or duration_us < 0:
                raise RuntimeError(
                    f"Invalid NPU kernel duration {duration_text!r} in "
                    f"{kernel_detail_files[0]}."
                )
            total_us += duration_us
            row_count += 1

    if row_count == 0:
        raise RuntimeError(
            f"No NPU kernel duration was recorded in {kernel_detail_files[0]}."
        )
    return total_us / 1000.0


def benchmark_npu_with_profiler(
    fn: Callable[[], None],
    *,
    prepare_fn: Optional[Callable[[], None]] = None,
    warmup: int = 2,
    rep: int = 3,
) -> float:
    """Benchmark ``fn`` with the NPU profiler and return median kernel time."""
    if warmup < 0 or rep <= 0:
        raise ValueError("warmup must be non-negative and rep must be positive")
    if getattr(_NPU_PROFILER_BENCHMARK_STATE, "active", False):
        raise RuntimeError(
            "Recursive scheduler NPU-profiler benchmark is not supported."
        )

    import torch_npu
    from torch._inductor.runtime.runtime_utils import cache_dir

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        l2_cache=False,
        data_simplification=False,
    )

    profile_parent = os.path.join(
        cache_dir(), "npu_scheduler_profile", f"{os.getpid()}_{uuid.uuid4().hex}"
    )
    sample_times_ms = []

    with _NPU_PROFILER_BENCHMARK_LOCK:
        if npu_profiler_session_active():
            raise RuntimeError(
                "Scheduler NPU-profiler benchmark cannot run inside an active "
                "torch_npu profiler session."
            )

        _NPU_PROFILER_BENCHMARK_STATE.active = True
        try:
            cache_buffer = (
                triton.runtime.driver.active.get_empty_cache_for_benchmark()
            )
            try:
                for _ in range(warmup):
                    if prepare_fn is not None:
                        prepare_fn()
                    triton.runtime.driver.active.clear_cache(cache_buffer)
                    fn()
                torch.npu.synchronize()

                for sample_idx in range(rep):
                    if prepare_fn is not None:
                        prepare_fn()
                    triton.runtime.driver.active.clear_cache(cache_buffer)
                    torch.npu.synchronize()

                    sample_path = os.path.join(
                        profile_parent, f"sample_{sample_idx}"
                    )
                    with torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.NPU],
                        schedule=torch_npu.profiler.schedule(
                            wait=0, warmup=0, active=1, repeat=1
                        ),
                        on_trace_ready=simple_trace_handler(sample_path),
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=False,
                        with_flops=False,
                        with_modules=False,
                        experimental_config=experimental_config,
                    ) as prof:
                        fn()
                        torch.npu.synchronize()
                        prof.step()

                    sample_times_ms.append(
                        _read_kernel_duration_sum_ms(sample_path)
                    )
            finally:
                shutil.rmtree(profile_parent, ignore_errors=True)
        finally:
            _NPU_PROFILER_BENCHMARK_STATE.active = False

    return statistics.median(sample_times_ms)


def _run_prepare(prepare_fn: Optional[Callable[[], None]]) -> None:
    if prepare_fn is not None:
        prepare_fn()


def benchmark_npu_with_mspti(
    fn: Callable[[], None],
    *,
    prepare_fn: Optional[Callable[[], None]] = None,
    warmup: int = 5,
    rep: int = 30,
) -> float:
    """Measure all kernels from one invocation with MSPTI, in milliseconds."""
    if warmup < 0 or rep <= 0:
        raise ValueError("warmup must be non-negative and rep must be positive")
    if KernelMonitor is None:
        raise ImportError("mspti package not found.")
    if "libmspti.so" not in os.getenv("LD_PRELOAD", ""):
        raise RuntimeError(
            "libmspti.so is not preloaded; start Python with libmspti.so in "
            "LD_PRELOAD to use the CV MSPTI benchmark backend."
        )
    if getattr(_NPU_PROFILER_BENCHMARK_STATE, "active", False):
        raise RuntimeError("Recursive scheduler MSPTI benchmark is not supported.")

    with _NPU_PROFILER_BENCHMARK_LOCK:
        if npu_profiler_session_active():
            raise RuntimeError(
                "Scheduler MSPTI benchmark cannot run inside an active "
                "torch_npu profiler session."
            )
        _NPU_PROFILER_BENCHMARK_STATE.active = True
        try:
            cache_buffer = triton.runtime.driver.active.get_empty_cache_for_benchmark()
            for _ in range(warmup):
                _run_prepare(prepare_fn)
                triton.runtime.driver.active.clear_cache(cache_buffer)
                fn()
            torch.npu.synchronize()

            sample_times_ms = []
            for _ in range(rep):
                _run_prepare(prepare_fn)
                triton.runtime.driver.active.clear_cache(cache_buffer)
                torch.npu.synchronize()
                durations_ns = []
                invalid_durations = []

                def callback(data):
                    duration_ns = data.end - data.start
                    if not math.isfinite(duration_ns) or duration_ns < 0:
                        invalid_durations.append(duration_ns)
                    else:
                        durations_ns.append(duration_ns)

                from mspti import MsptiResult
                monitor = KernelMonitor()
                start_result = monitor.start(callback)
                if start_result != MsptiResult.MSPTI_SUCCESS:
                    try:
                        monitor.stop()
                    finally:
                        raise RuntimeError(
                            f"Failed to start MSPTI KernelMonitor: {start_result}."
                        )
                try:
                    fn()
                    torch.npu.synchronize()
                finally:
                    stop_result = monitor.stop()
                if stop_result != MsptiResult.MSPTI_SUCCESS:
                    raise RuntimeError(
                        f"Failed to stop MSPTI KernelMonitor: {stop_result}."
                    )
                if invalid_durations:
                    raise RuntimeError(
                        f"Invalid MSPTI kernel durations: {invalid_durations!r}."
                    )

                if not durations_ns:
                    raise RuntimeError(
                        "MSPTI did not record an NPU kernel for the benchmark "
                        "invocation."
                    )
                sample_times_ms.append(sum(durations_ns) / 1_000_000.0)
        finally:
            _NPU_PROFILER_BENCHMARK_STATE.active = False

    return statistics.median(sample_times_ms)


def _split_marker_samples_ms(
    rows: list[dict[str, str]],
    *,
    start_marker_name: str,
    end_marker_name: str,
    expected_count: int,
    steps_per_entry: int,
    allow_zero: bool = False,
) -> list[float]:
    """Split profiler rows by marker kernels and sum each callable interval."""
    if expected_count <= 0 or steps_per_entry <= 0:
        raise ValueError("expected_count and steps_per_entry must be positive")

    def is_marker(name: str, marker: str) -> bool:
        name = name.lower()
        compact_name = name.replace("_", "")
        return name.endswith(marker) or compact_name.startswith(f"{marker}aicore")

    samples_us: list[list[float]] = [[] for _ in range(expected_count)]
    current_entry = 0
    current_sum_us = 0.0
    current_step = 0
    collecting = False

    for row in rows:
        name = (row.get("Name") or "").lower()
        if is_marker(name, start_marker_name):
            if collecting:
                raise RuntimeError("Profiler start marker appeared before end marker.")
            if current_entry >= expected_count:
                raise RuntimeError(
                    "Profiler marker count is larger than benchmark entry count."
                )
            current_sum_us = 0.0
            collecting = True
            continue
        if is_marker(name, end_marker_name):
            if not collecting:
                continue
            if current_entry >= expected_count:
                raise RuntimeError(
                    "Profiler marker count is larger than benchmark entry count."
                )
            samples_us[current_entry].append(current_sum_us)
            current_sum_us = 0.0
            collecting = False
            current_step += 1
            if current_step == steps_per_entry:
                current_step = 0
                current_entry += 1
            continue

        duration_text = (row.get("Duration(us)") or "").strip()
        if not duration_text:
            continue
        duration_us = float(duration_text)
        if not math.isfinite(duration_us) or duration_us < 0:
            raise RuntimeError(f"Invalid NPU kernel duration {duration_text!r}.")
        if collecting:
            current_sum_us += duration_us

    if collecting or current_entry != expected_count or current_step != 0:
        raise RuntimeError(
            "Profiler marker count does not match benchmark entry count."
        )

    timings_ms = []
    for entry_samples in samples_us:
        if len(entry_samples) != steps_per_entry:
            raise RuntimeError(
                "Profiler marker samples do not match benchmark step count."
            )
        timing_ms = statistics.median(entry_samples) / 1000.0
        if not allow_zero and timing_ms <= 0:
            raise RuntimeError(
                f"Profiler marker interval has no positive kernel time: {timing_ms}."
            )
        timings_ms.append(timing_ms)
    return timings_ms


def benchmark_npu_batch_with_profiler(
    entries: List[Tuple[Callable[[], None], Optional[Callable[[], None]]]],
    *,
    warmup: int = 2,
    rep: int = 6,
    allow_zero: bool = False,
) -> list[float]:
    """Batch benchmark callables with NPU profiler marker splitting."""
    if warmup < 0 or rep <= 0:
        raise ValueError("warmup must be non-negative and rep must be positive")
    if not entries:
        return []
    if getattr(_NPU_PROFILER_BENCHMARK_STATE, "active", False):
        raise RuntimeError(
            "Recursive scheduler NPU-profiler benchmark is not supported."
        )

    import torch_npu
    from torch._inductor.runtime.runtime_utils import cache_dir

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        l2_cache=False,
        data_simplification=False,
    )

    profile_path = os.path.join(
        cache_dir(), "npu_cv_profile", f"{os.getpid()}_{uuid.uuid4().hex}"
    )
    marker = torch.ones(1, dtype=torch.float32, device="npu")
    start_marker_name = "neg"
    end_marker_name = "abs"

    with _NPU_PROFILER_BENCHMARK_LOCK:
        if npu_profiler_session_active():
            raise RuntimeError(
                "Scheduler NPU-profiler benchmark cannot run inside an active "
                "torch_npu profiler session."
            )

        _NPU_PROFILER_BENCHMARK_STATE.active = True
        try:
            cache_buffer = triton.runtime.driver.active.get_empty_cache_for_benchmark()
            try:

                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(
                        wait=1 * len(entries),
                        warmup=1 * len(entries),
                        active=rep * len(entries),
                        repeat=1,
                    ),
                    on_trace_ready=simple_trace_handler(profile_path),
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                    with_flops=False,
                    with_modules=False,
                    experimental_config=experimental_config,
                ) as prof:
                    for fn, prepare_fn in entries:
                        for _ in range(rep + 2):
                            marker.neg_()
                            torch.npu.synchronize()
                            fn()
                            torch.npu.synchronize()
                            marker.abs_()
                            torch.npu.synchronize()
                            prof.step()

                kernel_detail_files = []
                for root, _, files in os.walk(profile_path):
                    for file_name in files:
                        if file_name == "kernel_details.csv":
                            kernel_detail_files.append(os.path.join(root, file_name))
                if len(kernel_detail_files) != 1:
                    raise RuntimeError(
                        "Expected exactly one kernel_details.csv under "
                        f"{profile_path}, found {len(kernel_detail_files)}."
                    )

                with open(
                    kernel_detail_files[0], newline="", encoding="utf-8"
                ) as csv_file:
                    reader = csv.DictReader(csv_file)
                    if "Name" not in (reader.fieldnames or ()) or "Duration(us)" not in (
                        reader.fieldnames or ()
                    ):
                        raise RuntimeError(
                            f"Profiler result {kernel_detail_files[0]} is missing "
                            "Name or Duration(us)."
                        )
                    rows = list(reader)
                return _split_marker_samples_ms(
                    rows,
                    start_marker_name=start_marker_name,
                    end_marker_name=end_marker_name,
                    expected_count=len(entries),
                    steps_per_entry=rep,
                    allow_zero=allow_zero,
                )
            finally:
                shutil.rmtree(profile_path, ignore_errors=True)
        finally:
            _NPU_PROFILER_BENCHMARK_STATE.active = False


def benchmark_npu_batch_with_mspti(
    entries: List[Tuple[Callable[[], None], Optional[Callable[[], None]]]],
    *,
    warmup: int = 5,
    rep: int = 30,
    allow_zero: bool = False,
) -> list[float]:
    """Batch benchmark callables with MSPTI marker splitting."""
    if warmup < 0 or rep <= 0:
        raise ValueError("warmup must be non-negative and rep must be positive")
    if not entries:
        return []
    if KernelMonitor is None:
        raise ImportError("mspti package not found.")
    if "libmspti.so" not in os.getenv("LD_PRELOAD", ""):
        raise RuntimeError(
            "libmspti.so is not preloaded; start Python with libmspti.so in "
            "LD_PRELOAD to use the CV MSPTI benchmark backend."
        )
    if getattr(_NPU_PROFILER_BENCHMARK_STATE, "active", False):
        raise RuntimeError("Recursive scheduler MSPTI benchmark is not supported.")

    with _NPU_PROFILER_BENCHMARK_LOCK:
        if npu_profiler_session_active():
            raise RuntimeError(
                "Scheduler MSPTI benchmark cannot run inside an active "
                "torch_npu profiler session."
            )

        _NPU_PROFILER_BENCHMARK_STATE.active = True
        try:
            cache_buffer = triton.runtime.driver.active.get_empty_cache_for_benchmark()
            marker = torch.empty(1, dtype=torch.float32, device="npu")
            start_marker_name = "neg"
            end_marker_name = "abs"
            rows: list[dict[str, str]] = []
            invalid_durations = []

            def callback(data):
                duration_ns = data.end - data.start
                if not math.isfinite(duration_ns) or duration_ns < 0:
                    invalid_durations.append(duration_ns)
                    return
                rows.append(
                    {
                        "Name": getattr(data, "name", ""),
                        "Duration(us)": str(duration_ns / 1000.0),
                    }
                )

            from mspti import MsptiResult

            monitor = KernelMonitor()
            start_result = monitor.start(callback)
            if start_result != MsptiResult.MSPTI_SUCCESS:
                try:
                    monitor.stop()
                finally:
                    raise RuntimeError(
                        f"Failed to start MSPTI KernelMonitor: {start_result}."
                    )
            try:
                for fn, prepare_fn in entries:
                    for _ in range(rep):
                        marker.neg_()
                        torch.npu.synchronize()
                        fn()
                        torch.npu.synchronize()
                        marker.abs_()
                        torch.npu.synchronize()
            finally:
                stop_result = monitor.stop()
            if stop_result != MsptiResult.MSPTI_SUCCESS:
                raise RuntimeError(
                    f"Failed to stop MSPTI KernelMonitor: {stop_result}."
                )
            if invalid_durations:
                raise RuntimeError(
                    f"Invalid MSPTI kernel durations: {invalid_durations!r}."
                )

            return _split_marker_samples_ms(
                rows,
                start_marker_name=start_marker_name,
                end_marker_name=end_marker_name,
                expected_count=len(entries),
                steps_per_entry=rep,
                allow_zero=allow_zero,
            )
        finally:
            _NPU_PROFILER_BENCHMARK_STATE.active = False


def _benchmark_npu_with_benchmarker(
    fn: Callable[[], None],
    *,
    prepare_fn: Optional[Callable[[], None]] = None,
) -> float:
    from torch._inductor.runtime.benchmarking import benchmarker

    def wrapped_fn():
        _run_prepare(prepare_fn)
        fn()

    return benchmarker.benchmark_gpu(wrapped_fn)


def benchmark_npu_for_cv(
    fn: Callable[[], None],
    prepare_fn: Optional[Callable[[], None]] = None,
    *,
    allow_zero: bool = False,
) -> float:
    """Benchmark one CV fusion callable with mspti -> profiler -> benchmarker."""
    get_npu_cv_benchmark_backend()
    errors: list[str] = []
    for backend in _CV_BENCHMARK_FALLBACK_BACKENDS:
        try:
            if backend == "mspti":
                ms = benchmark_npu_with_mspti(fn, prepare_fn=prepare_fn)
            elif backend == "batch_profile":
                ms = benchmark_npu_batch_with_profiler(
                    [(fn, prepare_fn)], allow_zero=allow_zero
                )[0]
            elif backend == "benchmarker":
                ms = _benchmark_npu_with_benchmarker(fn, prepare_fn=prepare_fn)
            else:
                raise AssertionError(f"unknown CV benchmark backend {backend}")
            log.info("[npu-cv-benchmark] backend=%s ms=%.6f", backend, ms)
            return ms
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
            log.warning(
                "[npu-cv-benchmark] backend=%s failed, trying next backend: %s",
                backend,
                exc,
            )
    raise RuntimeError(
        "All CV benchmark backends failed: " + " | ".join(errors)
    )


def benchmark_npu_for_cv_batch(
    entries: List[Tuple[Callable[[], None], Optional[Callable[[], None]]]],
) -> list[float]:
    """Benchmark CV selector choices with mspti -> profiler -> benchmarker."""
    get_npu_cv_benchmark_backend()
    if not entries:
        return []

    errors: list[str] = []
    for backend in _CV_BENCHMARK_FALLBACK_BACKENDS:
        try:
            if backend == "mspti":
                timings = benchmark_npu_batch_with_mspti(entries)
            elif backend == "batch_profile":
                timings = benchmark_npu_batch_with_profiler(entries)
            elif backend == "benchmarker":
                timings = [
                    _benchmark_npu_with_benchmarker(fn, prepare_fn=prepare_fn)
                    for fn, prepare_fn in entries
                ]
            else:
                raise AssertionError(f"unknown CV benchmark backend {backend}")
            if len(timings) != len(entries):
                raise RuntimeError(
                    f"{backend} returned {len(timings)} timings for "
                    f"{len(entries)} benchmark entries."
                )
            log.info(
                "[npu-cv-benchmark-batch] backend=%s count=%d",
                backend,
                len(timings),
            )
            return timings
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
            log.warning(
                "[npu-cv-benchmark-batch] backend=%s failed, trying next backend: %s",
                backend,
                exc,
            )
    raise RuntimeError(
        "All CV batch benchmark backends failed: " + " | ".join(errors)
    )


class SimpleProfilingAnalyzer:
    def __init__(self, output_dir: str = None, worker_name: str = None):
        self.output_dir = output_dir
        ProfPathCreator().init(dir_name=output_dir, worker_name=worker_name)
        self.op_summary_files = None
        self.op_statistic_files = None
        self.use_custom_analyzer = True

    def _get_msprof_py_script_path(self):
        from .._inductor.cpp_builder import get_ascend_home
        ascend_home_path = get_ascend_home()
        script_path = "tools/profiler/profiler_tool/analysis/msprof/msprof.py"
        full_script_path = os.path.join(ascend_home_path, script_path)
        if os.path.exists(full_script_path):
            return full_script_path
        else:
            return None

    def _msprof_py_export(self, msprof_py_script_path, ascend_pt_dir) -> None:
        if not (os.path.exists(msprof_py_script_path) and os.path.exists(ascend_pt_dir)):
            raise RuntimeError("Failed to run export subprocess, command or ascend_pt dir not found.")
        prof_dir = None
        for item in os.listdir(ascend_pt_dir):
            item_path = os.path.join(ascend_pt_dir, item)
            if os.path.isdir(item_path) and item.startswith('PROF'):
                prof_dir = item_path
                break
        if prof_dir is None:
            raise RuntimeError("PROF directory not found.")
        mindstudio_profiler_output_dir = os.path.join(prof_dir, "mindstudio_profiler_output")
        if os.path.exists(mindstudio_profiler_output_dir):
            shutil.rmtree(mindstudio_profiler_output_dir)
        export_cmd = [sys.executable, msprof_py_script_path, "export", "summary", "-dir", prof_dir]
        completed_analysis = subprocess.run(export_cmd, capture_output=True)
        if completed_analysis.returncode != 0:
            raise RuntimeError("subprocess return code is not 0.")
        op_summary_file_pattern = os.path.join(mindstudio_profiler_output_dir, "op_summary*.csv")
        op_summary_files = glob.glob(op_summary_file_pattern)
        self.op_summary_files = op_summary_files
        op_statistic_file_pattern = os.path.join(mindstudio_profiler_output_dir, "op_statistic*.csv")
        op_statistic_files = glob.glob(op_statistic_file_pattern)
        self.op_statistic_files = op_statistic_files
        if len(op_summary_files) == 0:
            raise RuntimeError("export results not found.")

    def _convert_op_summary_to_kernel_details(self):
        if len(self.op_summary_files) == 0:
            return
        ascend_profiler_output_dir = os.path.join(self.ascend_pt_dir, "ASCEND_PROFILER_OUTPUT")
        kernel_details_path = os.path.join(ascend_profiler_output_dir, "kernel_details.csv")
        os.makedirs(ascend_profiler_output_dir, exist_ok=True)
        if len(self.op_summary_files) == 1:
            header_replace_mapping = {',Op Name,': ',Name,', ',Task Duration(us),': ',Duration(us),'}
            with open(self.op_summary_files[0], 'r', encoding='utf-8') as op_summary_file:
                lines = op_summary_file.readlines()
                if lines:
                    ori_line0 = lines[0]
                    for ori_str, sub_str in header_replace_mapping.items():
                        lines[0] = lines[0].replace(ori_str, sub_str)
                    if lines[0] == ori_line0:
                        raise RuntimeError("convert op_summary to kernel_details failed, replace header failed.")
                    with open(kernel_details_path, 'w', encoding='utf-8') as kernel_details_file:
                        kernel_details_file.writelines(lines)
                else:
                    raise RuntimeError("convert op_summary to kernel_details failed, op_summary file empty.")
        else:
            import pandas as pd
            df_list = []
            for file in self.op_summary_files:
                df = pd.read_csv(file)
                df_list.append(df)

            merged_df = pd.concat(df_list, ignore_index=True)
            merged_df = merged_df.sort_values(by='Task ID')
            column_rename_mapping = {'Op Name': 'Name', 'Task Duration(us)': 'Duration(us)'}
            merged_df.rename(columns=column_rename_mapping, inplace=True)
            merged_df.to_csv(kernel_details_path, index=False, encoding="utf-8")

    def _convert_op_statistic(self):
        if len(self.op_statistic_files) == 0:
            return
        ascend_profiler_output_dir = os.path.join(self.ascend_pt_dir, "ASCEND_PROFILER_OUTPUT")
        op_statistic_path = os.path.join(ascend_profiler_output_dir, "op_statistic.csv")
        os.makedirs(ascend_profiler_output_dir, exist_ok=True)
        if len(self.op_statistic_files) == 1:
            shutil.copy(self.op_statistic_files[0], op_statistic_path)
        else:
            import pandas as pd
            df_list = []
            for file in self.op_statistic_files:
                df = pd.read_csv(file)
                df_list.append(df)
            merged_df = pd.concat(df_list, ignore_index=True)
            merged_df.to_csv(op_statistic_path, index=False, encoding="utf-8")

    def trace_ready(self, prof_inst):
        msprof_py_script_path = self._get_msprof_py_script_path()
        ascend_pt_dir = prof_inst.prof_if.prof_path
        self.ascend_pt_dir = ascend_pt_dir
        try:
            print_info_msg(f'Start parsing profiling data: {ascend_pt_dir}')
            export_start = datetime.now(tz=timezone.utc).astimezone()
            self._msprof_py_export(msprof_py_script_path, ascend_pt_dir)
            self._convert_op_summary_to_kernel_details()
            self._convert_op_statistic()
            export_end = datetime.now(tz=timezone.utc).astimezone()
            print_info_msg(f'SimpleProfilingAnalyzer parsed in a total time of {export_end - export_start}')
        except Exception as e:
            # fullback to default tensorboard_trace_handler
            print_warn_msg(f'Failed to run msprof.py subprocess, reason: {e}, fallback to tensorboard_trace_handler.')
            self.use_custom_analyzer = False
            prof_inst.prof_if.analyse(async_mode=False)


@no_exception_func()
def simple_trace_handler(dir_name: str = None, worker_name: str = None):
    prof_analyzer = SimpleProfilingAnalyzer(dir_name, worker_name)
    return prof_analyzer.trace_ready


def mspti_batch_benchmark(kernel_funcs: List[Callable], warmup=5, active=30,
                          profiling_clear_l2=False, filter_list=None):
    if KernelMonitor is None:
        raise ImportError('mspti package not found.')
    if 'libmspti.so' not in os.getenv('LD_PRELOAD', ''):
        raise RuntimeError('libmspti.so not set in LD_PRELOAD, please set libmspti.so in LD_PRELOAD to use mspti.')
    if profiling_clear_l2:
        cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    else:
        cache = None

    kernel_cnt = len(kernel_funcs)
    all_kernel_durations = []

    def callback(data):
        if profiling_clear_l2 and ('zero' in data.name.lower() or 'zeroslike' in data.name.lower()):
            return
        if filter_list and not any(x in data.name for x in filter_list):
            return
        all_kernel_durations.append(data.end - data.start)

    monitor = KernelMonitor()
    torch.npu.synchronize()
    monitor.start(callback)
    for fn in kernel_funcs:
        for _ in range(warmup + active):
            if profiling_clear_l2:
                cache.zero_()
            fn()
    torch.npu.synchronize()
    monitor.stop()

    if len(all_kernel_durations) != kernel_cnt * (warmup + active):
        raise RuntimeError("A call to kernel_func should result in exactly one kernel record."
                           "Consider use filter_list argument.")

    duration_per_kernel = []
    for _ in range(kernel_cnt):
        current_kernel_durations = all_kernel_durations[:warmup + active]
        all_kernel_durations = all_kernel_durations[warmup + active:]
        current_active_kernel_durations = current_kernel_durations[warmup:]
        # calculate the average kernel duration in microseconds.
        kernel_avg_duration = sum(current_active_kernel_durations) / len(current_active_kernel_durations) / 1000
        duration_per_kernel.append(kernel_avg_duration)
    return duration_per_kernel
