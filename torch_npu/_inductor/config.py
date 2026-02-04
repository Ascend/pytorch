import logging
import os  # noqa: C101
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
import torch
from torch._inductor import config
from triton.runtime.driver import driver
from torch_npu.npu._backends import get_soc_version

from .utils import classproperty

enable_npu_indexing = True

config.triton.unique_kernel_names = True
# avoid test_opensora_cases_model_16_forward  reinterpre_tensor issue
config.allow_buffer_reuse = False
# inductor debug switch
config.trace.enabled = True

config.fallback_random = True

# npu hardware params from trion
target = driver.active.get_current_target()
device = driver.active.get_current_device()
prop = driver.active.utils.get_device_properties(device)

num_cube_core = prop["num_aicore"]
num_vector_core = prop["num_aicore"]

Ascend910B1 = 220
Ascend310B1 = 240
Ascend910_9391 = 250
# unit byte 
npu_block = 32


# For debug
class aot_inductor:
    # If debug_kernel is set, codegen in python wrapper (output_code.py) and cpp wrapper (model.pt2)
    # will be modified to dump fx graph and weights. Meanwhile, generate repro func in output_code.py. 
    # Then, run aoti and output_code.py will dump tensor args before and after each triton kernel,
    # which can be used to detect which kernel is incorrect.
    debug_kernel = os.environ.get("AOTI_ASCEND_DEBUG_KERNEL", False)

    # No need to set debug_kernel_in_run manually. It will be set in output_code.py
    # by codegen if debug_kernel is set.
    debug_kernel_in_run = False

    # Path that to be used for dump weights in aoti to reproduce when debug_kernel is set.
    repro_tensor_path = os.environ.get("AOTI_ASCEND_REPRO_TENSOR_PATH", "aoti_repro_tensors")

    # Path that to be used for dump tensor args before and after triton kernel in aoti execute
    # when debug_kernel is set.
    dump_path_cpp = os.environ.get("AOTI_ASCEND_DUMP_PATH_CPP", "aoti_dump_cpp")

    # Path that to be used for dump tensor args before and after triton kernel in output_code.py
    # when debug_kernel_in_run is set.
    dump_path_py = os.environ.get("AOTI_DUMP_PATH_PY", "aoti_dump_py")


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

    # Wheter to ignore GEMM template for standard matmul
    catlass_ignore_gemm_in_standard_mm: bool = True

    catlass_epilogue_fusion_enable = (
        os.environ.get("CATLASS_EPILOGUE_FUSION", "0") == "1"
    )

    catlass_bench_use_profiling: bool = (
        os.environ.get("TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING", "0") == "1"
    )

    # Note: This fuction is not implemented yet.
    # enable generation of inline standalone runner in CATLASS CPP generated code
    # which allows to compile the generated code into a standalone executable.
    generate_test_runner: bool = (
        os.environ.get("INDUCTOR_NPU_BACKEND_GENERATE_TEST_RUNNER_CODE", "0") == "1"
    )

    catlass_enabled_ops: str = os.environ.get("TORCHINDUCTOR_CATLASS_ENABLED_OPS", "mm,addmm,bmm")


traced_fx_graph_cache = os.environ.get("INDUCTOR_ASCEND_FX_GRAPH_CACHE", None)
check_accuracy = os.environ.get("INDUCTOR_ASCEND_CHECK_ACCURACY", False)
auto_fallback = os.environ.get("INDUCTOR_ASCEND_AUTO_FALLBACK", True)
fallback_warning = os.environ.get("INDUCTOR_ASCEND_FALLBACK_WARNING", False)

# Trace fx graph when lowering and dump.
dump_fx_graph = os.environ.get("INDUCTOR_ASCEND_DUMP_FX_GRAPH", False) \
                or check_accuracy \
                or aot_inductor.debug_kernel
# Specify kernel ids that to be force fallback to fx graph call.
# Usage: `torch_npu._inductor.config.force_fallback_kernel_id = 'all' `
#    or  `torch_npu._inductor.config.force_fallback_kernel_id = [1, 2, 10] `
# (1) 'all' means try to fallback all kernel to fx graph call.
# (2) [1, 2, 10] means try to fallback kernel like triton_xxx_1, triton_xxx_2 and triton_xxx_10
force_fallback_kernel_id = []

acc_comp_tol = {
    torch.float32: {'rtol': 1.3e-6, 'atol': 1e-5},
    torch.float16: {'rtol': 1e-3, 'atol': 1e-5},
    torch.bfloat16: {'rtol': 1.6e-2, 'atol': 1e-5},
    "default": {'rtol': 1.3e-6, 'atol': 1e-5},
}

ub_size = 192 * 1024
if get_soc_version() >= Ascend910_9391:
    ub_size = 256 * 1024

if Ascend910B1 <= get_soc_version() < Ascend310B1 or get_soc_version() >= Ascend910_9391:
    num_vector_core = num_cube_core * 2

use_store_in_cat = os.environ.get("USE_STORE_IN_CAT", False)
max_cat_size_in_per_kernel = 4 * 1024
inductor_indirect_memory_mode = None
if get_soc_version() >= Ascend910_9391:
    # A5 INDUCTOR_INDIRECT_MEMORY_MODE: fallback, simt_template, simt_only, simd_simt_mix
    inductor_indirect_memory_mode = os.environ.get("INDUCTOR_INDIRECT_MEMORY_MODE", None)
    if os.environ.get("TRITON_EMBEDDING_FUSION", None):
        inductor_indirect_memory_mode = "simt_template"
    if os.environ.get("INDUCTOR_ASCEND_INDIRECT_MEMORY_SIMT_TEMPLATE", None) == "0":
        inductor_indirect_memory_mode = "simt_only"
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
default_nddma_switch = '1' if get_soc_version() >= Ascend910_9391 else '0'
nddma_switch = os.getenv("TORCHINDUCTOR_NDDMA", default_nddma_switch) == '1'

lowering_cat_with_concat_kernel = False
if get_soc_version() >= Ascend910_9391:
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


def set_compile_threads():
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        torchinductor_compile_threads = int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])
        if torchinductor_compile_threads == 1:
            return


def disable_comprehensive_padding():
    torch._inductor.config.comprehensive_padding = False