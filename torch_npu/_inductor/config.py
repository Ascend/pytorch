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

# npu hardware params from trion
target = driver.active.get_current_target()
device = driver.active.get_current_device()
prop = driver.active.utils.get_device_properties(device)

num_cube_core = prop["num_aicore"]
num_vector_core = prop["num_aicore"]

# unit byte
npu_block = 32

traced_fx_graph_cache = os.environ.get("INDUCTOR_ASCEND_FX_GRAPH_CACHE", None)
check_accuracy = os.environ.get("INDUCTOR_ASCEND_CHECK_ACCURACY", False)
auto_fallback = os.environ.get("INDUCTOR_ASCEND_AUTO_FALLBACK", True)
fallback_warning = os.environ.get("INDUCTOR_ASCEND_FALLBACK_WARNING", False)

acc_comp_tol = {
    torch.float32: {'rtol': 1.3e-6, 'atol': 1e-5},
    torch.float16: {'rtol': 1e-3, 'atol': 1e-5},
    torch.bfloat16: {'rtol': 1.6e-2, 'atol': 1e-5},
    "default": {'rtol': 1.3e-6, 'atol': 1e-5},
}

if ("Ascend910B" in target.arch):
    num_vector_core = num_cube_core * 2

log_level_env = os.getenv('INDUCTOR_ASCEND_LOG_LEVEL', 'ERROR').upper()
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
