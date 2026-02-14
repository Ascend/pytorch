import logging
import os  # noqa: C101
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
import torch
from torch._inductor import config
from triton.runtime.driver import driver

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

if ("Ascend910B" in target.arch):
    num_vector_core = num_cube_core * 2

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


def set_compile_threads():
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        torchinductor_compile_threads = int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])
        if torchinductor_compile_threads == 1:
            return
        log.warning(f"TORCHINDUCTOR_COMPILE_THREADS is set to {torchinductor_compile_threads}, "
                    "but currently only support 1. It will be modified to 1.")

    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    torch._inductor.config.compile_threads = 1

    def get_env_num_workers():
        return 1
    torch._inductor.select_algorithm.get_env_num_workers = get_env_num_workers


def disable_comprehensive_padding():
    torch._inductor.config.comprehensive_padding = False