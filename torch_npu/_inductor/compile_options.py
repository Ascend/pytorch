"""NPU compile options exposed as torch.compile(options={"npu.<key>": ...}).

Registered as the ``npu`` device's ``device_custom_config`` via
``register_npu_backend`` below, so its keys join the FX graph cache key;
register npu codegen backends only through it to keep ``npu.*`` routing.
"""

import sys
from typing import Optional

from torch.utils._config_module import Config, install_config_module


npu_backend: str = "default"
enable_shape_handling: bool = False
shape_handling_configs: list = []
shape_handling_dict: Optional[dict] = Config(default=None, value_type=dict)


# adds patch, save_config, etc
install_config_module(sys.modules[__name__])


def register_npu_backend(
    scheduling, wrapper_codegen, cpp_wrapper_codegen=None, backend_config=None
):
    """Register the npu codegen backend and its device config.

    backend_config: a backend's own ConfigModule (e.g. triton_experimental's
    ncfg). On torch without pytorch#194998 it keeps the device config slot,
    so its values stay in the FX graph cache key exactly as before. On torch
    with pytorch#194998 this module owns the slot for the "npu.*" option
    routing, and backend_config is registered under its own module name so
    the FX cache key snapshot and the out-of-process config transport (both
    per-module upstream) keep covering it.
    """
    # import inside the function: install_config_module above moves
    # module-level values of config types into _config, which would delete
    # the name from this module's globals.
    from torch._inductor.codegen.common import (
        custom_backend_codegen_configs,
        register_backend_for_device,
    )
    from torch_npu._compat.inductor import namespaced_compile_options

    if namespaced_compile_options():
        device_custom_config = sys.modules[__name__]
        if backend_config is not None and backend_config is not sys.modules[__name__]:
            # keeps the backend config in the registry under its own module
            # name: upstream collects and restores these per module
            custom_backend_codegen_configs[backend_config.__name__] = backend_config
    else:
        device_custom_config = backend_config
    register_backend_for_device(
        "npu",
        scheduling,
        wrapper_codegen,
        cpp_wrapper_codegen,
        device_custom_config=device_custom_config,
    )
