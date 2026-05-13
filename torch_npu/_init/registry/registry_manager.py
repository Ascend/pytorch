import torch
from torch.utils.checkpoint import DefaultDeviceType

import torch_npu


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
    from torch_npu.utils.npu_intercept import _cann_package_check

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


def _register_dynamo():
    """
    Register Dynamo integration:
    - Dynamo backend
    - Dynamo device interface
    - NPU trace rules for Dynamo
    """
    from torch_npu._init.registry.dynamo import (
        register_dynamo_backends,
        register_dynamo_device_interface,
        register_dynamo_trace_rules,
    )

    register_dynamo_backends()
    register_dynamo_device_interface()

    # Do not repeat this call for register_dynamo_trace_rules appends rules into
    # Dynamo's global rules maps.
    register_dynamo_trace_rules()


def _register_rpc():
    """
    Register and init RPC NPU backend.
    """
    from torch_npu.distributed.rpc.backend_registry import _rpc_backend_registry

    _rpc_backend_registry()


def _register_inductor():
    """
    Register lightweight NPU device op overrides for Inductor.
    Do not import toch_npu._inductor here: toch_npu._inductor performs full NPU
    Inductor backend loading and heavy global patches lazily when torch.compile
    and Inductor path is actually used.
    """
    from torch_npu.utils._inductor import _inductor_register_device_op_overrides

    _inductor_register_device_op_overrides()


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
    2. Distributed and Dynamo depend on NPU backend / _C children.
    3. RPC, dtensor and inductor are Python-side framework integrations.
    4. DefaultDeviceType is set after NPU backend is registered.
    """
    if not hasattr(torch_npu, "_C"):
        raise RuntimeError(
            "torch_npu._C is not available before torch_npu registry init"
        )

    _register_npu_backend()
    _register_distributed()
    _register_dynamo()
    _register_rpc()
    _register_inductor()
    _register_default_gradient_device_type()
