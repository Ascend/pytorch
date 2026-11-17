# pylint: disable=missing-docstring,import-outside-toplevel,wrong-import-position

import sys

__all__ = []

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

        def load_submodules(module):
            module._backend = importlib.import_module("fxrt.fx_backend").backend
            # Avoid loading the wrapper twice through its two import paths.
            importlib.import_module("fxrt.fx_wrapper")

        module = sys.modules.get("fxrt")
        if module is not None:
            load_submodules(module)
            return module
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
            load_submodules(module)
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
