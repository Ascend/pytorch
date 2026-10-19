"""Compatibility for older Triton Ascend host-extension builders."""

import functools
import inspect
import types

import torch
from torch.torch_version import TorchVersion


class _Cxx20Subprocess:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def run(self, args, *positional, **kwargs):
        # These builders pass an argv list, including the hard-coded standard.
        # Preserve newer standards and do not mutate the caller's command.
        if isinstance(args, (list, tuple)):
            args = [
                "-std=c++20" if flag in ("-std=c++17", "-std=gnu++17") else flag
                for flag in args
            ]
        return self.original.run(args, *positional, **kwargs)


def _with_cxx20(builder):
    # Clone only this function's globals: replacing subprocess.run globally
    # would affect unrelated compilations and race with compilation threads.
    namespace = dict(builder.__globals__)
    namespace["subprocess"] = _Cxx20Subprocess(namespace["subprocess"])
    patched = types.FunctionType(
        builder.__code__, namespace, builder.__name__, builder.__defaults__, builder.__closure__
    )
    patched.__kwdefaults__ = builder.__kwdefaults__
    signature = inspect.signature(builder)

    @functools.wraps(builder)
    def build(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments.get("kernel_launcher", "torch") != "torch":
            return builder(*args, **kwargs)
        return patched(*args, **kwargs)

    build._torch_npu_cxx20 = True
    return build


def patch_triton_extension_cxx_standard():
    if TorchVersion(torch.__version__) < "2.13":
        return
    try:
        from triton.backends.ascend import driver, utils
    except ImportError:
        return
    from triton.backends import backends

    namespaces = [vars(utils), vars(driver)]
    # Triton 3.2 discovers drivers using spec_from_file_location without adding
    # the module to sys.modules. Its registered class therefore uses a different
    # globals dict from the ordinarily imported ascend.driver module. Patch that
    # namespace too, without initializing the lazy runtime driver/NPU context.
    backend = backends.get("ascend")
    if backend is not None:
        namespace = getattr(backend.driver.__init__, "__globals__", None)
        if namespace is not None:
            namespaces.append(namespace)
    for namespace in namespaces:
        for name in ("_build_npu_ext", "_precompile_npu_ext"):
            original = namespace.get(name)
            if original is None or getattr(original, "_torch_npu_cxx20", False):
                continue
            namespace[name] = _with_cxx20(original)
