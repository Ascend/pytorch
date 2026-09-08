import torch

from torch_npu._compat.version import CURRENT_VERSION

# torch PR #192345 (first in torch 2.15) fires the backend's
# _dynamo_backend_init hook at backend resolution. Both the hook attach
# (torch >= 2.15) and the historical torch._dynamo.optimize fallback
# (torch < 2.15) live here, version-isolated.


def _npu_backend_init():
    """Eagerly initialize the NPU backend once "npu" is resolved.

    torch >= 2.15 calls this via the backend's ``_dynamo_backend_init`` hook
    ahead of the first graph capture. It fires on every backend resolution,
    so it must be idempotent.
    """
    from torch_npu.dynamo import _get_global_npu_backend

    _get_global_npu_backend("npu")


# COMPAT(>= 2.15): attach the hook only on torch >= 2.15. Skipped for the
#   eager fallback (no torchair): _npu_backend_init would raise.
# CAN REMOVE the version check when MIN_SUPPORTED >= (2, 15)
def compat_attach_dynamo_backend_init(backend):
    if CURRENT_VERSION < (2, 15):
        return

    from torch_npu.dynamo import _lazy_exec, _npu_backend_entrypoint

    if backend is not _lazy_exec:
        return

    _lazy_exec._dynamo_backend_init = _npu_backend_init
    _npu_backend_entrypoint._dynamo_backend_init = _npu_backend_init


# COMPAT(>= 2.15): torch < 2.15 has no _dynamo_backend_init hook, so replay
#   the historical torch._dynamo.optimize patch for compile-time eager init.
# CAN REMOVE this function when MIN_SUPPORTED >= (2, 15)
def compat_patch_dynamo_optimize():
    if CURRENT_VERSION >= (2, 15):
        return

    from torch import _TorchCompileWrapper
    from torch_npu.dynamo import _get_global_npu_backend

    if getattr(torch._dynamo.optimize, "__module__", None) == __name__:
        return

    src_optimize = torch._dynamo.optimize

    def npu_optimize(*args, **kwargs):
        backend = None
        if "backend" in kwargs:
            backend = kwargs["backend"]
        elif len(args) == 1:
            backend = args[0]

        backend_name = None
        if isinstance(backend, str):
            backend_name = backend
        elif isinstance(backend, _TorchCompileWrapper):
            backend_name = backend.compiler_name

        if backend_name == "npu":
            # Init torchair ahead of running model.
            _get_global_npu_backend(backend_name)
        return src_optimize(*args, **kwargs)

    torch._dynamo.optimize = npu_optimize
