__all__ = ["HiFloat8Tensor", "erase_stream", "matmul_checksum"]

import os


# Disable autoloading before running 'import torch' to avoid circular dependencies
ORG_AUTOLOAD = os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["TORCH_WARM_POOL"] = "0"

import torch


def _check_device_conflict():
    acc = torch._C._get_accelerator()
    if acc.type not in ("cpu", "npu"):
        import time

        # torch_npu.utils._error_code.ErrCode.NOT_SUPPORT
        error_code = "ERR00007"
        error_code_msg = "feature not supported"
        submodule_name = "PTA"
        raise RuntimeError(
            f"Two accelerators cannot be used at the same time "
            f"in PyTorch: npu and {acc.type}. You can install "
            f"the cpu version of PyTorch to use your npu device, "
            f"or use the {acc.type} device with "
            f"'export TORCH_DEVICE_BACKEND_AUTOLOAD=0'.\n"
            f"[ERROR] {time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} "
            f"(PID:{os.getpid()}, Device:-1, RankID:-1) "
            f"{error_code} {submodule_name} {error_code_msg}"
        )


_check_device_conflict()


# Import-time env access logging patch. Keep early to capture initialization-time getenv.
import torch_npu.utils.patch_getenv
from torch_npu._init.core.module_loader import _load_core_modules
from torch_npu._init.core.optional_features import _enable_optional_features
from torch_npu._init.core.runtime_lifecycle import _initialize_runtime_lifecycle
from torch_npu._init.patches.patch_manager import _apply_all_patches
from torch_npu._init.registry.registry_manager import _register_components
from torch_npu.version import __version__ as __version__


def _initialize():
    # 1. core modules, registration side effects and public API export
    _load_core_modules()

    # 2. backend and framework integration registration
    _register_components()

    # 3. apply patches
    _apply_all_patches()

    # 4. final extension barrier and shutdown hook
    _initialize_runtime_lifecycle()

    # 5. optional runtime features 
    _enable_optional_features()


_initialize()


# This function is an entrypoint called by PyTorch
# when running 'import torch'. There is no need to do anything.
def _autoload():
    # We should restore this switch as sub processes need to inherit its value
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = ORG_AUTOLOAD
