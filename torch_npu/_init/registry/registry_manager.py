import os
import torch
from torch.utils.checkpoint import DefaultDeviceType

import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error
from torch_npu.utils.collect_env import get_cann_version


cann_pytorch_version_map = {
    "6.3.RC2": ["1.8.1.post2", "1.11.0.post1", "2.0.0.rc1"],
    "6.3.RC1": ["1.8.1.post1", "1.11.0"],
    "6.1.RC1": ["1.8.1.post1", "1.11.0"],
    "6.0.1": ["1.8.1", "1.11.0.rc2"],
    "6.0.RC1": ["1.8.1", "1.11.0.rc1"]
}


def _cann_package_check():
    if "ASCEND_HOME_PATH" in os.environ:
        ascend_home_path = os.environ["ASCEND_HOME_PATH"]
        if not os.path.exists(ascend_home_path):
            raise Exception(f"ASCEND_HOME_PATH : {ascend_home_path} does not exist. "
                            "Please run 'source set_env.sh' in the CANN installation path." +
                            pta_error(ErrCode.NOT_FOUND))

        # check whether environment variables are correctly configured
        if "ASCEND_OPP_PATH" not in os.environ:
            raise Exception("ASCEND_OPP_PATH environment variable is not set. "
                            "Please check whether the opp package has been installed. If exist, please run "
                            "'source set_env.sh' in the CANN installation path." +
                            pta_error(ErrCode.NOT_FOUND))

        ascend_opp_path = os.environ["ASCEND_OPP_PATH"]
        if not os.path.exists(ascend_opp_path):
            raise Exception(f"ASCEND_OPP_PATH : {ascend_opp_path} does not exist. "
                            "Please check whether the opp package has been installed. If exist, please run "
                            "'source set_env.sh' in the CANN installation path." +
                            pta_error(ErrCode.NOT_FOUND))

        ascend_runtime_path = os.path.join(ascend_home_path, "runtime")
        if not os.path.exists(ascend_runtime_path):
            raise Exception(f"ASCEND_RUNTIME_PATH : {ascend_runtime_path} does not exist. "
                            "Please check whether the runtime package has been installed. If exist, please run "
                            "'source set_env.sh' in the CANN installation path." +
                            pta_error(ErrCode.NOT_FOUND))

        ascend_compiler_path = os.path.join(ascend_home_path, "compiler")
        if not os.path.exists(ascend_compiler_path):
            raise Exception(f"ASCEND_COMPILER_PATH : {ascend_compiler_path} does not exist. "
                            "Please check whether the compiler package has been installed. If exist, please run "
                            "'source set_env.sh' in the CANN installation path." +
                            pta_error(ErrCode.NOT_FOUND))

        # get the cann version
        cann_version = get_cann_version()

        # check whether the CANN package version matches the pytorch version
        if cann_version in cann_pytorch_version_map and \
                torch_npu.__version__ not in cann_pytorch_version_map[cann_version]:
            print(f"Warning: CANN package version {cann_version} and PyTorch version {torch_npu.__version__} "
                  "do not match. Please check the README of the Ascend PyTorch repo.")
    else:
        print("Warning: ASCEND_HOME_PATH environment variable is not set.")


def _register_npu_backend():
    """
    Register core NPU backend capability:
    - privateuse1 backend
    - torch.npu device module
    - Tensor / Module / Storage npu methods
    - CANN package / environment check

    Note:
    This function must not initialize NPU runtime.
    NPU runtime initialization is ownde by torch_npu.npu._lazy_init().
    """
    from torch_npu._init.registry.backend import register_privateuse1_backend

    register_privateuse1_backend()
    _cann_package_check()

    if not hasattr(torch, "npu"):
        raise RuntimeError(
            "torch.npu is not registered after privateuse1 backend registration"
        )

def _register_distributed():
    """
    Register distributed backend for NPU.

    Dependency:
    - _C._distributed_c10d must be ready.
    - distributed runtime should have been initialized by ModuleLoader.
    """
    if not hasattr(torch_npu._C, "_distributed_c10d"):
        raise RuntimeError(
            "torch_npu._C._distributed_c10d must be ready before distributed backend registration"
        )

    from torch_npu._init.registry.distributed import (
        register_distributed_backend_for_npu,
    )

    # init and register distributed backend
    register_distributed_backend_for_npu()

def _register_rpc():
    """
    Register and init RPC NPU backend.
    """
    from torch_npu.distributed.rpc.backend_registry import _rpc_backend_registry

    _rpc_backend_registry()


def _register_default_gradient_device_type():
    """
    Set default device type for gradient checkpointing.
    """
    DefaultDeviceType.set_device_type("npu")

def _register_components():
    """
    Register torch_npu backend and integration capabilities.

    Order matters:
    1. NPU backend is the base capability.
    2. Distributed depends on NPU backend / _C children.
    3. RPC is a Python-side framework integration.
    4. DefaultDeviceType is set after NPU backend is registered.
    """
    if not hasattr(torch_npu, "_C"):
        raise RuntimeError(
            "torch_npu._C is not available before torch_npu registry init"
        )

    _register_npu_backend()
    _register_distributed()
    _register_rpc()
    _register_default_gradient_device_type()
