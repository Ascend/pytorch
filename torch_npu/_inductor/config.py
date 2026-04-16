import logging
import os  # noqa: C101
import re
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import torch
from torch._inductor import config
import torch._inductor.config as inductor_config

from torch_npu.npu._backends import get_soc_version

from .utils import classproperty


# By default, native Torch/inductor set 'inplace_buffers = True', while it will disable NPU-IR's multi-buffer.
# Here, we add this env variable as a switch to decide whether or not to reuse a kernel input as its output.
enable_inplace_buffers = os.environ.get('ENABLE_INPLACE_BUFFERS', '1').lower() in ('1', 'true', 'yes')
if not enable_inplace_buffers:
    inductor_config.inplace_buffers = False

# inductor debug switch
config.trace.enabled = True

config.fallback_random = True

device = torch.npu.current_device()
prop = torch.npu.get_device_properties(device)

num_cube_core = prop.cube_core_num
num_vector_core = prop.vector_core_num

Ascend910B1 = 220
Ascend310B1 = 240
Ascend910_9391 = 250
Ascend950 = 260
is_ascend950 = get_soc_version() >= Ascend950



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
        os.environ.get("TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING", "0") == "1"
    )

    # Note: This function is not implemented yet.
    # enable generation of inline standalone runner in CATLASS CPP generated code
    # which allows to compile the generated code into a standalone executable.
    generate_test_runner: bool = (
        os.environ.get("INDUCTOR_NPU_BACKEND_GENERATE_TEST_RUNNER_CODE", "0") == "1"
    )

    catlass_enabled_ops: str = os.environ.get("TORCHINDUCTOR_CATLASS_ENABLED_OPS", "mm,addmm,bmm")


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

traced_fx_graph_cache = os.environ.get("INDUCTOR_ASCEND_FX_GRAPH_CACHE", None)
check_accuracy = os.environ.get("INDUCTOR_ASCEND_CHECK_ACCURACY", False)
auto_fallback = os.environ.get("INDUCTOR_ASCEND_AUTO_FALLBACK", True)
fallback_warning = os.environ.get("INDUCTOR_ASCEND_FALLBACK_WARNING", False)

# Trace fx graph when lowering and dump.
dump_fx_graph = os.environ.get("INDUCTOR_ASCEND_DUMP_FX_GRAPH", False) \
                or check_accuracy


def parse_rtol_atol(env_str: str):
    rtol, atol = None, None
    if not env_str.strip():
        return rtol, atol
    
    parts = [p.strip() for p in env_str.split(",") if p.strip()]
    for part in parts:
        match = re.match(r"^(rtol|atol)\s*=\s*([0-9.eE+-]+)$", part, re.IGNORECASE)
        if not match:
            logging.warning(f"INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL environment variable has invalid format: {part}. "
                            f"It should be like 'rtol=1e-6,atol=1e-5'.")
            continue
        
        key, value_str = match.groups()
        try:
            value = float(value_str)
            if key.lower() == "rtol":
                rtol = value
            elif key.lower() == "atol":
                atol = value
        except ValueError:
            logging.warning(f"INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL environment variable has invalid value for {key}: {value_str}. "
                            f"It should be a float number.")
            continue
    
    return rtol, atol

# Default threshold
rtol_f32 = 1.3e-6
rtol_f16 = 1e-3
rtol_bf16 = 1.6e-2
rtol_default = 1.3e-6 
atol_default = 1e-5

if dump_fx_graph:
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

ub_size = 192 * 1024
if is_ascend950:
    ub_size = 256 * 1024

if Ascend910B1 <= get_soc_version() < Ascend310B1 or get_soc_version() >= Ascend910_9391:
    num_vector_core = num_cube_core * 2

use_store_in_cat = os.environ.get("USE_STORE_IN_CAT", False)
max_cat_size_in_per_kernel = 4 * 1024
max_cat_count_in_per_kernel = None
inductor_indirect_memory_mode = None
if is_ascend950:
    # A5 INDUCTOR_INDIRECT_MEMORY_MODE: fallback, simt_template, simt_only, simd_simt_mix
    inductor_indirect_memory_mode = os.environ.get("INDUCTOR_INDIRECT_MEMORY_MODE", "simd_simt_mix")
    if inductor_indirect_memory_mode == "fallback":
        inductor_indirect_memory_mode = None
    if inductor_indirect_memory_mode not in [None, "simt_template", "simt_only", "simd_simt_mix"]:
        inductor_indirect_memory_mode = "simd_simt_mix"
    # if mode in "simt_only", "simd_simt_mix", should use load store cat
    if inductor_indirect_memory_mode in ["simt_only", "simd_simt_mix"]:
        use_store_in_cat = True
        # simt only or simd_simt_mix only need small size for concat
        max_cat_size_in_per_kernel = 1024

# simt default stacksize is 256 * 32 Byte
simt_default_warp_stacksize = 256 * 32

# nddma switch
default_nddma_switch = '1' if is_ascend950 else '0'
nddma_switch = os.getenv("TORCHINDUCTOR_NDDMA", default_nddma_switch) == '1'

lowering_cat_with_concat_kernel = False
if is_ascend950:
    lowering_cat_with_concat_kernel = True

log_level_env = os.getenv('INDUCTOR_ASCEND_LOG_LEVEL', 'WARNING').upper()
log_level_mapping = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
log_level = log_level_mapping.get(log_level_env.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

aggresive_autotune = os.getenv("INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE", '0').lower() in ('1', 'true')
inductor_static_mode = os.environ.get('INDUCTOR_STATIC_MODE', '0').lower() in ('1', 'yes', 'true')
profile_path = "./profile_result/"

fasta_autotune = os.environ.get('FASTAUTOTUNE', "0") == "1"
fasta_autotune_method = os.getenv("AUTOTUNE_METHOD", "Expert")
if fasta_autotune:
    if os.environ.get("ENABLE_PRINT_UB_BITS", "0") == "0":
        log.warning("Please set ENABLE_PRINT_UB_BITS to 1. Fasta autotune need to know real ub usage.")
        os.environ["ENABLE_PRINT_UB_BITS"] = "1"

    if fasta_autotune_method == "SampleStack" and torch._inductor.config.compile_threads != 1:
        log.warning(f"fasta SampleStack method is not temporarily compatible with multi-process compile, "
                    f"fasta_autotune set TORCHINDUCTOR_COMPILE_THREADS "
                    f"from {torch._inductor.config.compile_threads} to 1.")
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        torch._inductor.config.compile_threads = 1

max_precompiled_thread_num = os.cpu_count() // 2 # default precompile max thread num is half of the cpu count
if "TORCHNPU_PRECOMPILE_THREADS" in os.environ:
    max_precompiled_thread_num = int(os.environ["TORCHNPU_PRECOMPILE_THREADS"])

lowering_axis_count = None

def disable_comprehensive_padding():
    torch._inductor.config.comprehensive_padding = False