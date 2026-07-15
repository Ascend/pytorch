import ast
from enum import Enum, auto
import functools
import time

from torch_npu._inductor.fasta_autotune import log
from torch._inductor.codecache import _load_triton_kernel_from_source
from torch._inductor.runtime.triton_heuristics import NoTritonConfigsError
from torch_npu._inductor.runtime.triton_heuristics import stats



class CompilationPhase(Enum):
    """
    Representing the dynamic-filter compilation lifecycle
    the phases define how Triton kernel configs are compiled and benchmarked -
    when FASTA_DYNAMIC_FILTER is enabled

    Phases:
    R1: compile an initial subset of configs in async_compile (pre-run)
            benchmark in kernel.run()
    R2: iteratively compile and benchmark config batches from remaining configs
    DONE: final phase after the best launcher has been selected

    State transitions:
    R1 -> R2 -> DONE
    """
    R1 = auto()
    R2 = auto()
    DONE = auto()

    def next_phase(self):
        transitions = {
            CompilationPhase.R1: CompilationPhase.R2,
            CompilationPhase.R2: CompilationPhase.DONE,
            CompilationPhase.DONE: CompilationPhase.DONE,
        }
        return transitions[self]


class DynamicFilterScheduler:
    """
    Manages the autotuning workflow for FASTA_DYNAMIC_FILTER=1, including R1/R2
    autotuning, compilation, benchmarking, and schedule iterative refinement in R2.
    """
    def __init__(self, fasta_setting):
        if fasta_setting.autotune_method != "Expert":
            raise ValueError("dynamic filter is not supported in non-Expert autotune_method")

        self.phase = CompilationPhase.R1
        self.Rx = None
        self._selector = None
        self.phase_profiling_values = {}
        self.best_launchers = {}
        self.best_profiling_values = {}
        self.current_r2_configs = []
        self.precompile_time_s = 0.0

    def _build_minimal_kernel_src(self, filename: str) -> str:
        """
        Extracts a minimal triton kernel source from a python file,
        used to reload kernel from main process
        it keeps only the kernel function definition and required imports
        and removes unrelated code to reduce compilation overhead

        Args:
        filename (str): path to the python file containing the triton kernel

        Returns:
        str: minimal kernel source code as a string.
        """
        with open(filename) as f:
            src = f.read()
        tree = ast.parse(src)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        fn.decorator_list = [d for d in fn.decorator_list if "triton.jit" in ast.unparse(d)]
        used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}

        imports = []
        for node in tree.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = [
                a.asname or a.name.split(".")[0]
                for a in node.names
            ]
            if any(n in used for n in names):
                imports.append(node)

        module = ast.Module(body=imports + [fn], type_ignores=[])
        return ast.unparse(module)

    def prepare_r1_configs(self, fast_autotuner):
        if self.phase is CompilationPhase.R1:
            self.Rx = "R1"
            from .experimental.dynamic_filter.dynamic_filter_if import DynamicFilter as _SelectorCls
            self._selector = _SelectorCls(fast_autotuner.configs, fast_autotuner.get_fn_name())
            fast_autotuner.configs = self._selector.r1_configs
            log.info(f"r1_configs: {len(self._selector.r1_configs)}")

    def select_best_launcher_key(self, best_profiling_values, kernel_name):
        log.info(
            f"kernel: {kernel_name} - best launcher profiling time per batch: "
            f"{best_profiling_values}"
        )
        valid = {k: v for k, v in best_profiling_values.items() if v is not None}
        if not valid:
            raise RuntimeError("No valid launchers found")
        return min(valid, key=valid.get)

    def store_phase_profiling_values(self, fast_autotuner):
        if fast_autotuner.profile_values:
            self.phase_profiling_values[self.Rx] = fast_autotuner.profile_values
        else:
            self.phase_profiling_values[self.Rx] = []
        fast_autotuner.profile_values = None

    def compile_r2(self, fast_autotuner):
        """
        Compiles r2 configs during kernel.run()
        builds minimal kernel source if needed and reload it,

        Args:
        fast_autotuner: NPUFastAutotuner kernel wrapper object
        """
        fast_autotuner.skip_precompile = False
        load_kernel = None

        if getattr(fast_autotuner.fn, 'fn', None) is None:
            fn_src = self._build_minimal_kernel_src(filename=fast_autotuner.filename)
            kernel_name = fast_autotuner.fn.__name__
            load_kernel = functools.partial(_load_triton_kernel_from_source, kernel_name, fn_src)

        fast_autotuner.clear_last_record()
        fast_autotuner.configs = self.current_r2_configs
        fast_autotuner.profiling_config_num += len(fast_autotuner.configs)
        log.info(f"kernel: {fast_autotuner.get_fn_name()} - Compiling {self.Rx}, {len(fast_autotuner.configs)} configs.")

        fast_autotuner.precompile(warm_cache_only=False, reload_kernel=load_kernel)

    def log_autotune_timing_summary(self, kernel_name, _total_wall_s, r1_wall_s, refine_acum):
        _sel = self._selector
        _sel_stats = _sel.stats if _sel else {}
        _n_failed = len(_sel.not_profiled_indices) if _sel else 0
        _n_measured = len(_sel._measured) if _sel else 0
        _sel_overhead = _sel.overhead_ms if _sel else 0.0
        log.info(
            f"[FASTA_SUMMARY] kernel={kernel_name} "
            f"N={_sel_stats.get('N','?')} "
            f"D={_sel_stats.get('D','?')} "
            f"r1_size={len(_sel._r1_indices) if _sel else '?'} "
            f"measured={_n_measured} "
            f"failed={_n_failed} "
            f"savings={_sel_stats.get('savings_pct',0):.1f}% "
            f"rounds={_sel_stats.get('rounds_used','?')} "
            f"path={_sel_stats.get('path','?')} "
            f"mode={_sel_stats.get('mode','?')} "
            f"best_dur={_sel_stats.get('best_dur','?')} "
            f"total_wall={_total_wall_s*1000:.1f}ms "
            f"r1_wall={r1_wall_s*1000:.1f}ms "
            f"refine_total={refine_acum*1000:.1f}ms "
            f"selector_overhead={_sel_overhead:.2f}ms "
        )

    def store_precompile_duration(self, precompile_time_s):
        if self.phase is CompilationPhase.R1:
            self.precompile_time_s = precompile_time_s

    def compile_and_benchmark(self, fast_autotuner, *args, **kwargs):
        """
        Main entry point for the dynamic-filter compilation and benchmarking flow

        Depending on the current phase:
        - R1: initial benchmarking and transitions to R2
        - R2: iteratively compiles and benchmarks remaining config batches,
              progression and early stopping are controlled by dynamic_filter_algo
        - DONE: reuses the cached best launcher

        Args:
        fast_autotuner: NPUFastAutotuner kernel wrapper object
        """
        if self.phase is CompilationPhase.R1:
            autotune_start_time = time.perf_counter()
            self.autotuner(fast_autotuner, *args, **kwargs)
            r1_wall_s = time.perf_counter() - autotune_start_time
            kernel_name = fast_autotuner.get_fn_name()

            self.store_phase_profiling_values(fast_autotuner)
            self.phase = self.phase.next_phase()
            refine_acum = 0
            r1_pvs = self.phase_profiling_values.get(self.Rx, [])
            self._selector.update_batch_indices(r1_pvs)
            r1_timings = [pv.profiler_time for pv in r1_pvs]
            batch_configs = self._selector.refine(r1_timings)

            i = 0
            while batch_configs:
                self.Rx = f"{self.phase.name}_{i}"
                i = i + 1

                log.info(f"[FASTA_R2] {kernel_name} {self.Rx}: "
                            f"batch={len(batch_configs)} ")

                self.current_r2_configs = batch_configs
                self.autotuner(fast_autotuner, *args, **kwargs)

                self.store_phase_profiling_values(fast_autotuner)
                pvs = self.phase_profiling_values.get(self.Rx, [])
                self._selector.update_batch_indices(pvs)

                timings = [pv.profiler_time for pv in pvs]
                log.info(f"{kernel_name} b4 refine of {self.Rx} "
                            f"batch_configs:{len(batch_configs)} timings:{len(timings)}")

                t0 = time.perf_counter()
                batch_configs = self._selector.refine(timings)
                refine_acum += time.perf_counter() - t0

            _total_wall_s = time.perf_counter() - autotune_start_time
            self.log_autotune_timing_summary(kernel_name, _total_wall_s, r1_wall_s, refine_acum)

            self.phase = self.phase.next_phase()

            best_launcher_group = self.select_best_launcher_key(self.best_profiling_values, kernel_name)
            fast_autotuner.best_launcher = self.best_launchers.get(best_launcher_group)
            fast_autotuner.best_profiling_value = self.best_profiling_values.get(best_launcher_group)
            fast_autotuner.launchers = [fast_autotuner.best_launcher]
            log.info(
                f"{kernel_name} - "
                f"best config: {fast_autotuner.best_launcher.config.kwargs}, "
                f"from {best_launcher_group}, "
                f"cost time:{fast_autotuner.best_profiling_value}"
            )

            self.autotune_time_taken_ns = (_total_wall_s + self.precompile_time_s) * 1e9
            if fast_autotuner.save_cache_hook:
                fast_autotuner.save_cache_hook(fast_autotuner.launchers[0].config, self.autotune_time_taken_ns)

        if fast_autotuner.best_launcher is not None:
            fast_autotuner.launchers = [fast_autotuner.best_launcher]
            return

    def autotuner(self, fast_autotuner, *args, **kwargs):
        """
        Executes autotuning for the current phase
        and stores the best launcher per phase

        Args:
        fast_autotuner: NPUFastAutotuner kernel wrapper object
        """
        if self.phase is CompilationPhase.R2:
            self.compile_r2(fast_autotuner)
            if not fast_autotuner.launchers and not fast_autotuner.compile_results:
                self.best_launchers[self.Rx] = None
                log.warning(f"{self.Rx} produced no valid launchers")
                return

        if stats.enabled:
            fast_autotuner.cache_hit = False
        fast_autotuner.skip_precompile = False
        best_launcher = fast_autotuner.auto_tune_by_fasta_parallel(*args, **kwargs)

        fast_autotuner.best_launcher = best_launcher
        self.best_launchers[self.Rx] = fast_autotuner.best_launcher
        self.best_profiling_values[self.Rx] = fast_autotuner.best_profiling_value

    def catch_no_valid_triton_configs(self, kernel_name: str, e: NoTritonConfigsError):
        """
        Suppress error if no valid Triton configs are found in R2_x
        """
        if self.phase is CompilationPhase.R2:
            log.info(f"kernel: {kernel_name} - {self.Rx} has no valid configs\n, {e}")
            return
        raise e

    def get_compilation_desc(self):
        if self.phase is CompilationPhase.R1:
            return f"Precompile configs - {self.Rx}"
        if self.phase is CompilationPhase.R2:
            return f"Compile configs in run - {self.Rx}"
        return "Precompile configs"

    def get_benchmark_desc(self):
        if self.Rx is not None:
            return f"Benchmark configs - {self.Rx}"
        return "Benchmark configs"