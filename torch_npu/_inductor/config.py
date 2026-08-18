import logging
import os  # noqa: C101
import re
import sys
import math
from typing import Optional

import torch
import torch._inductor.config as inductor_config
from torch_npu.npu._backends import get_soc_version

from .utils import classproperty


# init inductor log
def _init_inductor_log():
    log_level_env = os.getenv("INDUCTOR_ASCEND_LOG_LEVEL", "WARNING").upper()
    log_level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = log_level_mapping.get(log_level_env, logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)

log = _init_inductor_log()


Ascend910B1 = 220
Ascend310B1 = 240
Ascend910_9391 = 250
Ascend950 = 260
is_ascend950 = get_soc_version() >= Ascend950

ub_size = 192 * 1024
if is_ascend950:
    ub_size = 256 * 1024


def _obtain_and_limit_cube_vector_core_num():
    # by default, obtain cube and vector core num from device properties
    device = torch.npu.current_device()
    prop = torch.npu.get_device_properties(device)
    cube_core_num, vector_core_num = prop.cube_core_num, prop.vector_core_num
    vector_cube_ratio = vector_core_num // cube_core_num
    log.info(
        "[_obtain_and_limit_cube_vector_core_num] obtain from device properties, "
        "cube_core_num=%s, vector_core_num=%s",
        cube_core_num, vector_core_num,
    )

    # obtain cube and vector core num from env (if env is set)
    npu_device_limit = os.environ.get("NPU_DEVICE_LIMIT", "")
    orig_npu_device_limit = npu_device_limit
    if npu_device_limit.strip():
        parts = [p.strip() for p in npu_device_limit.split(",") if p.strip()]
        if len(parts) != 2:
            log.error(
                "NPU_DEVICE_LIMIT=%r, which has invalid format:, "
                "It should be like '14,28' (cube_core_num,vector_core_num)",
                orig_npu_device_limit,
            )
            sys.exit(1)
        else:
            cube_str, vector_str = parts
            try:
                parsed_cube = int(cube_str)
                parsed_vector = int(vector_str)
            except ValueError:
                log.error(
                    "NPU_DEVICE_LIMIT=%r, which has invalid value, "
                    "Both cube_core_num and vector_core_num must be integers.",
                    orig_npu_device_limit,
                )
                sys.exit(1)
            else:
                if parsed_cube <= 0 or parsed_vector <= 0:
                    log.error(
                        "NPU_DEVICE_LIMIT=%r, which has non-positive value, "
                        "Both cube_core_num and vector_core_num must be positive value.",
                        orig_npu_device_limit,
                    )
                    sys.exit(1)
                if parsed_cube > cube_core_num or parsed_vector > vector_core_num:
                    log.error(
                        "NPU_DEVICE_LIMIT=%r, both cube_core_num and vector_core_num must "
                        "be less than or equal to device properties (%s, %s).",
                        orig_npu_device_limit, cube_core_num, vector_core_num,
                    )
                    sys.exit(1)
                if parsed_vector != parsed_cube * vector_cube_ratio:
                    log.error(
                        "NPU_DEVICE_LIMIT=%r, vector_core_num should be cube_core_num * %s",
                        orig_npu_device_limit, vector_cube_ratio,
                    )
                    sys.exit(1)

                cube_core_num = parsed_cube
                vector_core_num = parsed_vector
                log.info(
                    "[_obtain_and_limit_cube_vector_core_num] NPU_DEVICE_LIMIT from env: "
                    "cube_core_num=%s, vector_core_num=%s.",
                    cube_core_num, vector_core_num,
                )

    log.info(
        "[_obtain_and_limit_cube_vector_core_num] finished, "
        "cube_core_num=%s, vector_core_num=%s",
        cube_core_num, vector_core_num,
    )
    return prop, cube_core_num, vector_core_num

prop, num_cube_core, num_vector_core = _obtain_and_limit_cube_vector_core_num()


def _obtain_precompile_thread_num() -> int:
    """
    in torch/inductor/config.py, compile_threads is introduced with default value or
    by env TORCHINDUCTOR_COMPILE_THREADS; here, we obtain precompile_thread_num
    via default_value or env TORCHNPU_PRECOMPILE_THREADS.
    """
    # by default, inductor_config.compile_threads = 32 in torch/inductor/config.py
    precompile_thread_num = os.cpu_count() // max(inductor_config.compile_threads, 2)
    # by default, we set maximum of precompile_thread_num = 32
    precompile_thread_num = max(precompile_thread_num, 32)

    thread_num_str = os.environ.get("TORCHNPU_PRECOMPILE_THREADS", "")
    if thread_num_str.strip():
        try:
            precompile_thread_num = int(thread_num_str.strip())
        except ValueError as e:
            log.error(
                "TORCHNPU_PRECOMPILE_THREADS=%s with wrong value, %s", thread_num_str, e,
            )
            sys.exit(1)

    log.info(
        "for torch/inductor, compile_threads=%s; "
        "for torch_npu/inductor, precompile_thread_num=%s",
        inductor_config.compile_threads, precompile_thread_num,
    )
    return precompile_thread_num

precompile_thread_num = _obtain_precompile_thread_num()


# By default, native Torch/inductor set 'inplace_buffers = True', while it will disable NPU-IR's multi-buffer.
# Here, we add this env variable as a switch to decide whether or not to reuse a kernel input as its output.
enable_inplace_buffers = os.environ.get("ENABLE_INPLACE_BUFFERS", "1").lower() in (
    "1",
    "true",
    "yes",
)
if not enable_inplace_buffers:
    inductor_config.inplace_buffers = False

# inductor debug switch
inductor_config.trace.enabled = True

inductor_config.triton.coalesce_tiling_analysis = False
inductor_config.triton.mix_order_reduction = False

enable_fast_gelu = os.getenv("TORCHINDUCTOR_ENABLE_FAST_GELU", "0") == "1"
enable_flex_attention_dq_before_scale_materialize = os.environ.get(
    "FLEX_ATTENTION_DQ_BEFORE_SCALE_MATERIALIZE", "1"
).lower() in ("1", "true", "yes")


def _read_env_bool(name: str, default: str = "False") -> bool:
    value = os.environ.get(name, default)
    return value.strip().lower() in ("1", "true", "yes", "on")


# Enable the SIMT Welford lowering for variance and layer normalization.
# Keep it disabled by default while the new path is being rolled out.
enable_welford = os.getenv("TORCHINDUCTOR_ENABLE_WELFORD", "0") == "1"


class catlass:
    # Whether to enable debug info, e.g., line number
    enable_debug_info: bool = False

    @classproperty
    def catlass_dir(self) -> str:
        return os.environ.get(
            "TORCHINDUCTOR_NPU_CATLASS_DIR",
            os.path.abspath(
                os.path.join(os.path.dirname(torch.__file__), "../third_party/catlass")
            ),
        )

    # Configures the maximum number of CATLASS configs to profile in max_autotune.
    # By default it's None, so that all CATLASS configs are tuned.
    # This is mainly used to reduce test time in CI.
    catlass_max_profiling_configs: Optional[int] = None

    catlass_backend_min_gemm_size: int = 1

    # Whether to ignore GEMM template for standard matmul
    catlass_ignore_gemm_in_standard_mm: bool = True

    catlass_epilogue_fusion_enable = (
        os.environ.get("CATLASS_EPILOGUE_FUSION", "0") == "1"
    )

    catlass_bench_use_profiling: bool = (
        os.environ.get("TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING", "0")
        == "1"
    )

    # Note: This function is not implemented yet.
    # enable generation of inline standalone runner in CATLASS CPP generated code
    # which allows to compile the generated code into a standalone executable.
    generate_test_runner: bool = (
        os.environ.get("INDUCTOR_NPU_BACKEND_GENERATE_TEST_RUNNER_CODE", "0") == "1"
    )

    catlass_enabled_ops: str = os.environ.get(
        "TORCHINDUCTOR_CATLASS_ENABLED_OPS", "mm,addmm,bmm"
    )


class _npugraph_trees:
    def __init__(self):
        # skip cpu node check, eg: npu_fusion_attention_v3
        self._disable_cpu_input_check = False

    @property
    def disable_cpu_input_check(self):
        return self._disable_cpu_input_check

    @disable_cpu_input_check.setter
    def disable_cpu_input_check(self, value):
        self._disable_cpu_input_check = bool(value)
        # When disable_cpu_input_check is True, set slow_path_cudagraph_asserts to True to skip the CPU check.
        if value:
            torch._inductor.config.triton.slow_path_cudagraph_asserts = False


npugraph_trees = _npugraph_trees()

# NPU_INDUCTOR_FALLBACK_ALL=1 forces ops entering the NPU inductor lowering
# path to register fallback lowerings, so optimized/fused lowerings are not
# used. User-defined Triton kernel wrappers are still allowed to keep
# handwritten kernels runnable.
enable_full_lowering_fallback = os.environ.get("NPU_INDUCTOR_FALLBACK_LIST", "")
traced_fx_graph_cache = os.environ.get("INDUCTOR_ASCEND_FX_GRAPH_CACHE", None)
check_accuracy = os.environ.get("INDUCTOR_ASCEND_CHECK_ACCURACY", False)
auto_fallback = os.environ.get("INDUCTOR_ASCEND_AUTO_FALLBACK", True)
fallback_warning = os.environ.get("INDUCTOR_ASCEND_FALLBACK_WARNING", False)

# Trace fx graph when lowering and dump.
dump_fx_graph = os.environ.get("INDUCTOR_ASCEND_DUMP_FX_GRAPH", False) or check_accuracy


def parse_rtol_atol(env_str: str):
    rtol, atol = None, None
    if not env_str.strip():
        return rtol, atol

    parts = [p.strip() for p in env_str.split(",") if p.strip()]
    for part in parts:
        match = re.match(r"^(rtol|atol)\s*=\s*([0-9.eE+-]+)$", part, re.IGNORECASE)
        if not match:
            log.warning(
                "INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL environment variable has invalid format: %s. "
                "It should be like 'rtol=1e-6,atol=1e-5'.",
                part,
            )
            continue

        key, value_str = match.groups()
        try:
            value = float(value_str)
            if key.lower() == "rtol":
                rtol = value
            elif key.lower() == "atol":
                atol = value
        except ValueError:
            log.warning(
                "INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL environment variable has invalid value for %s: %s. "
                "It should be a float number.",
                key,
                value_str,
            )
            continue

    return rtol, atol


# Default threshold
rtol_f32 = 1.3e-6
rtol_f16 = 1e-3
rtol_bf16 = 1.6e-2
rtol_default = 1.3e-6
atol_default = 1e-5

if dump_fx_graph:
    torch._inductor.config.split_reductions = False
    # Configure accuracy comparison thresholds when check_accuracy is enabled
    ENV_TOL_STR = os.environ.get("INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL", "")
    rtol_custom, atol_custom = parse_rtol_atol(ENV_TOL_STR)

    if rtol_custom is not None:
        rtol_f32 = rtol_f16 = rtol_bf16 = rtol_default = rtol_custom
    if atol_custom is not None:
        atol_default = atol_custom

acc_comp_tol = {
    torch.float32: {"rtol": rtol_f32, "atol": atol_default},
    torch.float16: {"rtol": rtol_f16, "atol": atol_default},
    torch.bfloat16: {"rtol": rtol_bf16, "atol": atol_default},
    "default": {"rtol": rtol_default, "atol": atol_default},
}

inductor_indirect_memory_mode = None
if is_ascend950:
    # A5 INDUCTOR_INDIRECT_MEMORY_MODE: fallback, simt_template, simt_only, simd_simt_mix
    inductor_indirect_memory_mode = os.environ.get(
        "INDUCTOR_INDIRECT_MEMORY_MODE", "simd_simt_mix"
    )
    if inductor_indirect_memory_mode == "fallback":
        inductor_indirect_memory_mode = None
    if inductor_indirect_memory_mode not in [
        None,
        "simt_template",
        "simt_only",
        "simd_simt_mix",
    ]:
        inductor_indirect_memory_mode = "simd_simt_mix"

# simt default stacksize is 256 * 32 Byte
simt_default_warp_stacksize = 256 * 32

# nddma switch
default_nddma_switch = "1" if is_ascend950 else "0"
nddma_switch = os.getenv("TORCHINDUCTOR_NDDMA", default_nddma_switch) == "1"

aggresive_autotune = os.getenv("INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE", "0").lower() in (
    "1",
    "true",
)
enable_symbolic_shape_group_autotune = os.getenv(
    "INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE", "0"
).lower() in ("1", "true", "yes")

# Temporary rollout-only switch. Remove after grouped autotune is fully enabled.
symbolic_group_allow_templates = tuple(
    x.strip()
    for x in os.getenv(
        "INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES",
        "pointwise,reduction,persistent_reduction",
    ).split(",")
    if x.strip()
)

profile_path = "./profile_result/"

fasta_autotune = os.environ.get("FASTAUTOTUNE", "0") == "1"
fasta_autotune_method = os.getenv("AUTOTUNE_METHOD", "Expert")


def _parse_bool_env(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on", "y")


def _parse_float_env(name: str, default: float = 0.25, min_value: float = 0.0, max_value: float = 1.0) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        parsed = float(val)
    except (TypeError, ValueError):
        log.warning("Invalid %s=%r, fallback to %s", name, val, default)
        return default
    if parsed <= min_value or parsed > max_value:
        log.warning("Invalid %s=%r (must be in (%s, %s]), fallback to %s", name, val, min_value, max_value, default)
        return default
    return parsed


# Frontend -> inductor controls (env-driven)
# - TORCHINDUCTOR_NPU_FAST_LAUNCH: enable planned fast launch for NPU Python Wrapper
# - INDUCTOR_ASCEND_ENABLE_COSTMODEL: whether to forward costmodel backend signal to triton-ascend
# - INDUCTOR_ASCEND_COSTMODEL_RATIO: select the shortest-latency top ratio of configs
enable_fast_launch = _parse_bool_env(
    "TORCHINDUCTOR_NPU_FAST_LAUNCH",
    False,
)
enable_costmodel_backend = _parse_bool_env("INDUCTOR_ASCEND_ENABLE_COSTMODEL", False)
costmodel_ratio = _parse_float_env("INDUCTOR_ASCEND_COSTMODEL_RATIO", 0.25, 0.0, 1.0)
symbolic_group_max_benchmark_memory_ratio = _parse_float_env(
    "INDUCTOR_ASCEND_SYMBOLIC_GROUP_MAX_BENCHMARK_MEMORY_RATIO",
    0.25,
    0.0,
    1.0,
)


lowering_axis_count = None
inductor_ascend_linear_mode = "linear"

autotune_continue_on_failure = os.environ.get('TORCHINDUCTOR_NPU_BACKEND') == "default"

# fused_matmul_relu_pass: fuse relu(addmm) into npu_fused_matmul (A5-only op).
# Still under validation, so it is off by default; set the env var to 1 to enable.
enable_fused_matmul_relu = _parse_bool_env(
    "TORCHINDUCTOR_ENABLE_FUSED_MATMUL_RELU", False
)

# multi_slice_concat_pass: rewrite a run of constant-offset column slices feeding
# one cat into npu_ext::multi_slice_concat, so a single kernel replaces the N Slice
# copies aclnnCat needs. Gated off until validated on hardware; the rewrite is pure
# data movement and cannot affect numerics.
enable_multi_slice_concat = _parse_bool_env(
    "TORCHINDUCTOR_ENABLE_MULTI_SLICE_CONCAT", False
)

# grouped_matmul_fusion_pass: merge the independent small GEMMs feeding one cat into
# npu_grouped_matmul. Gated off until validated on the target model; the rewrite keeps
# every GEMM's operands intact but the kernel accumulates differently, so results are
# close but not bit-exact.
enable_grouped_matmul_fusion = _parse_bool_env(
    "TORCHINDUCTOR_ENABLE_GROUPED_MATMUL_FUSION", False
)

# permute_continous_reduction: when enabled, detects the "permute contiguous reduction"
# pattern (a non-reduction axis sitting between two reduction axes in stride order)
# and applies special handling: selects the permute axis as a tiling axis, uses NDDMA
# for non-contiguous data access, and merges the two reduction axes into one dimension.
# This optimizes kernels like LayerNorm-after-permute where data layout is transposed.
permute_continous_reduction = _parse_bool_env("PERMUTE_CONTINOUS_REDUCTION", False)

FLEX_ATTENTION_NPU_COMPILE_HINT_KEYS = (
    "limit_auto_multi_buffer_buffer",
    "multibuffer",
    "unit_flag",
    "enable_ubuf_saving",
    "hfusion_enable_multiple_consumer_fusion",
    "enable_select_analysis",
    "limit_auto_multi_buffer_only_for_local_buffer",
    "limit_auto_multi_buffer_of_local_buffer",
    "set_workspace_multibuffer",
    "tile_mix_vector_loop",
    "tile_mix_cube_loop",
    "enable_dynamic_cv_pipeline",
    "intra_cache_num",
    "inter_cache_num",
    "enable_cross_if_fusion",
    "enable_buffer_insert_optimization",
    "enable_ub_refine_opt",
)


class flex_attention:
    enable_npu_optimization = False
    use_config_generator = True
    metadata_auto_infer = True
    flexattention_mask_out = True
    # Keep rollout disabled until generated outputcode and NPU numerics have
    # been reviewed. Unsupported graphs always retain the legacy dK/dV path.
    bwd_dkdv_tasklist = True

    multibuffer = True
    unit_flag = True
    enable_ubuf_saving = True
    limit_auto_multi_buffer_buffer = "no-limit"
    hfusion_enable_multiple_consumer_fusion = True
    enable_select_analysis = False
    limit_auto_multi_buffer_only_for_local_buffer = False
    limit_auto_multi_buffer_of_local_buffer = "no-limit"
    set_workspace_multibuffer = 4
    tile_mix_vector_loop = 4
    tile_mix_cube_loop = 4
    enable_dynamic_cv_pipeline = False

    bwd_dq_limit_auto_multi_buffer_of_local_buffer = "no-l0c"
    bwd_dkdv_limit_auto_multi_buffer_of_local_buffer = "no-l0c"

    enable_buffer_insert_optimization = False
    enable_ub_refine_opt = False

    @classmethod
    def _filter_compile_options_for_soc(cls, options: dict) -> dict:
        options = options.copy()
        if is_ascend950:
            options.pop("enable_dynamic_cv_pipeline", None)
        return options

    @classmethod
    def _compile_options(cls, keys, overrides: Optional[dict] = None) -> dict:
        options = {key: getattr(cls, key) for key in keys}
        if overrides:
            options.update(overrides)
        return cls._filter_compile_options_for_soc(options)

    @classmethod
    def get_npu_compile_hint_params(cls) -> dict:
        return cls._compile_options(FLEX_ATTENTION_NPU_COMPILE_HINT_KEYS)

    @classmethod
    def get_sparse_mask_cvpipeline_compile_options(
        cls,
        *,
        enabled: bool,
        tile_mix_loop: int,
        enable_compile_hint: bool,
    ) -> dict:
        return cls._compile_options(
            (
                "enable_ubuf_saving",
                "unit_flag",
                "set_workspace_multibuffer",
                "limit_auto_multi_buffer_buffer",
                "hfusion_enable_multiple_consumer_fusion",
            ),
            overrides={
                "multibuffer": enabled,
                "limit_auto_multi_buffer_only_for_local_buffer": not enabled,
                "tile_mix_vector_loop": tile_mix_loop,
                "tile_mix_cube_loop": tile_mix_loop,
                "ENABLE_COMPILE_HINT": enable_compile_hint if enabled else False,
                "intra_cache_num": 3,
                "inter_cache_num": 2,
                "enable_cross_if_fusion": True,
                "enable_buffer_insert_optimization": True,
                "enable_ub_refine_opt": True,
            },
        )

    @classmethod
    def get_bwd_dq_compile_options(cls) -> dict:
        return cls._compile_options(
            (
                "limit_auto_multi_buffer_buffer",
                "hfusion_enable_multiple_consumer_fusion",
                "enable_select_analysis",
            ),
            overrides={
                "limit_auto_multi_buffer_of_local_buffer": (
                    cls.bwd_dq_limit_auto_multi_buffer_of_local_buffer
                ),
                "intra_cache_num": 3,
                "inter_cache_num": 2,
            },
        )

    @classmethod
    def get_bwd_dkdv_compile_options(cls) -> dict:
        return cls._compile_options(
            (
                "limit_auto_multi_buffer_buffer",
                "hfusion_enable_multiple_consumer_fusion",
                "unit_flag",
                "enable_dynamic_cv_pipeline",
            ),
            overrides={
                "limit_auto_multi_buffer_of_local_buffer": (
                    cls.bwd_dkdv_limit_auto_multi_buffer_of_local_buffer
                ),
                "intra_cache_num": 2,
                "inter_cache_num": 1,
            },
        )


flex_attention.bwd_dkdv_tasklist = _read_env_bool(
    "TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_BWD_DKDV_TASKLIST",
    "1" if flex_attention.bwd_dkdv_tasklist else "0",
)
flex_attention.flexattention_mask_out = _read_env_bool(
    "TORCHINDUCTOR_FLEXATTENTION_MASKOUT",
    "1" if flex_attention.flexattention_mask_out else "0",
)


def apply_flex_attention_npu_params(config: dict, *, enable: bool) -> dict:
    config = config.copy()
    if enable:
        config.update(flex_attention.get_npu_compile_hint_params())
        config["ENABLE_COMPILE_HINT"] = True
    else:
        config["ENABLE_COMPILE_HINT"] = False
    return config
