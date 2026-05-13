import importlib
import inspect
import os
import sys

import torch_npu
from torch_npu._init.core._exports import _export_public_apis


_REQUIRED_C_EXTENSION_CHILDREN = [
    "_cd",
    "_logging",
    "_flops_count",
    "_profiler",
    "_distributed_c10d",
]


def _register_c_extension_submodules(module, memo=None):
    """
    Mirror torch/__init__.py behavior:
    expose nested extension modules, e.g. torch_npu._C._distributed_c10d,
    through sys.modules so that Python import machinery can resolve them.
    """
    if memo is None:
        memo = set()
    if module in memo:
        return
    memo.add(module)

    module_name = module.__name__
    for name in dir(module):
        member = getattr(module, name)
        member_name = getattr(member, "__name__", "")
        if inspect.ismodule(member) and member_name.startswith(module_name):
            sys.modules.setdefault(member_name, member)
            _register_c_extension_submodules(member, memo)


def _create_child_once(_C, child_attr: str, init_method: str):
    if hasattr(_C, child_attr):
        return
    fn = getattr(_C, init_method, None)
    if callable(fn):
        fn()


def _initialize_c_extension_children(required_children):
    """
    Create and expose torch_npu._C child submodules.
    Every child submodule should be created here exactly once.
    Business Python modules must only consume them, not create them.
    """
    import torch_npu._C as _C  # ensure torch_npu._C is imported

    # Fixed order for child-module creation.
    _create_child_once(_C, "_profiler", "_profiler_init")
    _create_child_once(_C, "_distributed_c10d", "_c10d_npu_init")
    _create_child_once(_C, "_cd", "_cd_init")
    _create_child_once(_C, "_logging", "_logging_init")
    _create_child_once(_C, "_flops_count", "_flops_count_init")

    # Optional RPC child, only if built.
    _create_child_once(_C, "_distributed_rpc", "_rpc_npu_init")

    _register_c_extension_submodules(_C)
    missing = [name for name in required_children if not hasattr(_C, name)]
    if missing:
        raise RuntimeError(
            f"Required torch_npu._C child submodules are missing before Python import: {missing}"
        )


def _initialize_logging_if_needed():
    """
    Initialize logging runtime after _C._logging is ready.
    """
    if not hasattr(torch_npu._C, "_logging"):
        raise RuntimeError("torch_npu._C._logging is not initialized")

    from torch_npu._logging._internal import (
        _add_logging_module,
        _logging_patch,
        _update_log_state_from_env,
    )

    _logging_patch()
    _add_logging_module()
    _update_log_state_from_env()


def _initialize_profiler_if_needed():
    """
    Initialize profiler by enabling non-intrusive profiling hooks after _C._profiler is ready.
    """
    if not hasattr(torch_npu._C, "_profiler"):
        raise RuntimeError("torch_npu._C._profiler is not initialized")

    from torch_npu.profiler._non_intrusive_profile import _NonIntrusiveProfile

    _NonIntrusiveProfile.init()


def _initialize_rendezvous_if_needed():
    """
    Initialize rendezvous after distributed C-extension support is ready.
    """
    from torch_npu.distributed import is_available, rendezvous
    from torch_npu.utils._error_code import dist_error, ErrCode

    if is_available() and not hasattr(torch_npu._C, "_distributed_c10d"):
        raise RuntimeError(
            "torch_npu._C._distributed_c10d is not initialized"
            + dist_error(ErrCode.INTERNAL)
        )

    rendezvous._rendezvous_init()


def _check_npu_import():
    """
    Probe-import torch_npu.npu to convert environment issues (e.g. missing
    libhccl.so / libascendcl.so) into friendlier errors.
    Must be called only after _C / required child submodules / torch.npu are ready.
    """
    try:
        import torch_npu.npu  # noqa: F401
    except ImportError as e:
        from torch_npu.utils._error_code import ErrCode, pta_error

        if "libhccl.so" in str(e):
            if "ASCEND_OPP_PATH" in os.environ:
                # Warning: key logs in the fault mode library!!! Don't make arbitrary modifications!!!
                e.msg += (
                    ". Please check that the compiler package is installed. "
                    "Please run 'source set_env.sh' in the CANN installation path."
                    + pta_error(ErrCode.NOT_FOUND)
                )
            else:
                # Warning: key logs in the fault mode library!!! Don't make arbitrary modifications!!!
                e.msg += (
                    ". Please check that the cann package is installed. "
                    "Please run 'source set_env.sh' in the CANN installation path."
                    + pta_error(ErrCode.NOT_FOUND)
                )
        elif "libascendcl.so" in str(e):
            # Warning: key logs in the fault mode library!!! Don't make arbitrary modifications!!!
            e.msg += (
                ". Please check that the runtime package is installed. "
                "Please run 'source set_env.sh' in the CANN installation path."
                + pta_error(ErrCode.NOT_FOUND)
            )
        raise


def _load_core_modules():
    """
    Load torch_npu core modules.

    Includes:
    1. C extension child modules creation and exposure.
    2. Python runtime core support initialization:
        logging / profiler / distributed runtime.
    3. Check npu backend.
    4. Python registration modules imported for side effects.
    5. Public API export.

    Dependency:
    - Runtime support must run after _C child modules are ready.
    """
    _initialize_c_extension_children(required_children=_REQUIRED_C_EXTENSION_CHILDREN)

    # Do not hide these in another wrapper. Keep dependencies explicit.
    _initialize_logging_if_needed()
    _initialize_profiler_if_needed()
    _initialize_rendezvous_if_needed()

    _check_npu_import()

    _load_registration_modules()
    _export_public_apis()


def _load_registration_modules():
    """
    Import Python modules that rely on import-time side effects or old top-level submodule availability.

    Dependency:
    - Must run after core module loading.
    - _C child modules must already be ready.

    Includes:
    - ACLNN backend config module
    - old-compatible submodule imports: torch_npu.optim, torch_npu._afd
    - AFD op bindings: torch.ops.npu.<afd_op> -> torch_npu._afd.<afd_op>
    - custom ops import
    - op-plugin registration / meta registration / generated docs side effects
    """
    import torch_npu._afd  # noqa: F401
    import torch_npu.npu.aclnn  # noqa: F401
    import torch_npu.op_plugin
    import torch_npu.optim  # noqa: F401
    from torch_npu.op_plugin.meta import _meta_registrations  # noqa: F401
    from torch_npu.utils import custom_ops  # noqa: F401
    from torch_npu.utils._afd_ops import initialize_afd_bindings

    importlib.import_module("torch_npu._op_plugin_docs")
    if hasattr(torch_npu, "_op_plugin_docs"):
        delattr(torch_npu, "_op_plugin_docs")

    initialize_afd_bindings()
