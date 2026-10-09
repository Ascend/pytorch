import torch

from torch_npu._compat._impl import compat_impl, compat_impl_container

__all__ = [
    "patch_user_defined_class_variable",
    "compat_attach_dynamo_backend_init",
    "compat_patch_dynamo_optimize",
]

# upstream pytorch#191513 (first in torch 2.14) makes PyTorch discover Stream/Event
#   classes from registered DeviceInterface implementations. Older releases still
#   need torch_npu to add the NPU classes to UserDefinedClassVariable by hand.
#   The consumer (torch_npu/utils/_dynamo.py) imports this name and calls it.
@compat_impl(key="patch_user_defined_class_variable", ge=(2, 14))
def patch_user_defined_class_variable_upstream():
    """torch >= 2.14 discovers Stream/Event from DeviceInterface: nothing to do."""
    return


@compat_impl(key="patch_user_defined_class_variable", lt=(2, 14))
def patch_user_defined_class_variable_local():
    import functools

    from torch._dynamo.variables.user_defined import UserDefinedClassVariable

    original_method = UserDefinedClassVariable._in_graph_classes

    @staticmethod
    @functools.lru_cache(None)
    def patched_in_graph_classes():
        result = original_method()
        result.add(torch.npu.Event)
        result.add(torch.npu.Stream)
        return result

    UserDefinedClassVariable._in_graph_classes = patched_in_graph_classes


patch_user_defined_class_variable = compat_impl_container[
    "patch_user_defined_class_variable"
].resolve()


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


@compat_impl(key="compat_attach_dynamo_backend_init", ge=(2, 15))
def compat_attach_dynamo_backend_init_upstream(backend):
    """Attach the hook on torch >= 2.15.

    Skipped for the eager fallback (no torchair): _npu_backend_init would
    raise.
    """
    from torch_npu.dynamo import _lazy_exec, _npu_backend_entrypoint

    if backend is not _lazy_exec:
        return

    _lazy_exec._dynamo_backend_init = _npu_backend_init
    _npu_backend_entrypoint._dynamo_backend_init = _npu_backend_init


@compat_impl(key="compat_attach_dynamo_backend_init", lt=(2, 15))
def compat_attach_dynamo_backend_init_local(backend):
    """torch < 2.15 has no _dynamo_backend_init hook: do nothing."""
    return


compat_attach_dynamo_backend_init = compat_impl_container[
    "compat_attach_dynamo_backend_init"
].resolve()


@compat_impl(key="compat_patch_dynamo_optimize", ge=(2, 15))
def compat_patch_dynamo_optimize_upstream():
    """torch >= 2.15 has the hook: wrapping torch._dynamo.optimize is not needed."""
    return


@compat_impl(key="compat_patch_dynamo_optimize", lt=(2, 15))
def compat_patch_dynamo_optimize_local():
    """torch < 2.15 has no _dynamo_backend_init hook, so replay the historical
    torch._dynamo.optimize patch for compile-time eager init.
    """
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


compat_patch_dynamo_optimize = compat_impl_container[
    "compat_patch_dynamo_optimize"
].resolve()
