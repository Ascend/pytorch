# pylint: disable=missing-docstring,import-outside-toplevel,wrong-import-position

import sys

if __name__ != "fxrt":
    # fxrt ships as torch_npu.fxrt, and nothing else in torch_npu imports it, so
    # it is loaded only when torch_npu.fxrt is imported. Its modules import each
    # other as fxrt.*, and CPython's fromlist handling rebuilds submodule names
    # from __name__, so the package has to execute under the top-level name
    # "fxrt". This file is therefore loaded again under that name, the loaded
    # modules are mirrored as torch_npu.fxrt.*, and the "fxrt" module replaces
    # this one, so both import styles resolve to a single module tree.
    def _load_as_fxrt():
        import importlib.util
        import os

        if sys.modules.get("fxrt") is not None:
            return sys.modules["fxrt"]
        package_dir = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "fxrt",
            os.path.join(package_dir, "__init__.py"),
            submodule_search_locations=[package_dir],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["fxrt"] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for name in [n for n in sys.modules if n == "fxrt" or n.startswith("fxrt.")]:
                del sys.modules[name]
            raise
        return module

    _fxrt = _load_as_fxrt()
    for _name, _module in list(sys.modules.items()):
        if _name.startswith("fxrt."):
            sys.modules.setdefault(f"{__name__}.{_name[len('fxrt.'):]}", _module)
    sys.modules[__name__] = _fxrt
else:
    # fxrt ships only inside the torch_npu wheel, so it carries torch_npu's version.
    from torch_npu.version import __version__

    # The torch.compile backend: torch.compile(fn, backend=fxrt.backend).
    from fxrt.fx_backend import backend

    # Importing fxrt must not take over inductor's device codegen on its own, so the
    # fx_wrapper is registered only when the caller asks for it via
    # fxrt.register_fx_wrapper().
    from fxrt.fx_wrapper import register_fx_wrapper

    from fxrt import ops

    __all__ = ['backend', 'ops', 'register_fx_wrapper', '__version__']
