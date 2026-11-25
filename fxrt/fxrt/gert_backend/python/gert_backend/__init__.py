"""GE/OM2 backend shipped inside torch_npu.fxrt.

The ABI=1 producer exports a versioned Python dictionary to the isolated ABI=0
converter. Native converter libraries live only in this subpackage and are
loaded when compiling with this backend.
"""

import importlib
import importlib.abc
import importlib.util
from importlib.machinery import PathFinder
import sys


if __name__ != "fxrt.gert_backend":
    # Execute the package only under its canonical name. In particular, the
    # native extension must always pass through the loader installed below.
    sys.modules[__name__] = importlib.import_module("fxrt.gert_backend")
else:
    import ctypes
    import os

    _ALIAS_PACKAGE = "torch_npu.fxrt.gert_backend"
    _NATIVE_MODULE = "fxrt.gert_backend._fxrt_fx_converter"
    _GERT_IMPORT_ERROR = "GERT mode is not supported."
    _native_library = None

    class _NativeModuleLoader(importlib.abc.Loader):
        def __init__(self, loader):
            self.loader = loader

        def create_module(self, spec):
            try:
                global _native_library
                _native_library = ctypes.CDLL(spec.origin, mode=os.RTLD_NOW | os.RTLD_LOCAL)
                return self.loader.create_module(spec)
            except (ImportError, OSError, RuntimeError):
                raise ImportError(_GERT_IMPORT_ERROR) from None

        def exec_module(self, module):
            try:
                self.loader.exec_module(module)
            except (ImportError, OSError, RuntimeError):
                raise ImportError(_GERT_IMPORT_ERROR) from None

    class _GertAliasLoader(importlib.abc.Loader):
        def __init__(self, canonical_name):
            self.canonical_name = canonical_name

        def create_module(self, spec):
            module = importlib.import_module(self.canonical_name)
            self.canonical_spec = module.__spec__
            return module

        def exec_module(self, module):
            # module_from_spec overwrites __spec__ on the existing module.
            # Keep the canonical spec for reload and introspection.
            module.__spec__ = self.canonical_spec

    class _GertModuleFinder(importlib.abc.MetaPathFinder):
        _gert_package = __name__

        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith(_ALIAS_PACKAGE + "."):
                canonical_name = self._gert_package + fullname[len(_ALIAS_PACKAGE):]
                spec = importlib.util.find_spec(canonical_name)
                if spec is None:
                    return None
                alias_spec = importlib.util.spec_from_loader(
                    fullname,
                    _GertAliasLoader(canonical_name),
                    origin=spec.origin,
                    is_package=spec.submodule_search_locations is not None,
                )
                alias_spec.submodule_search_locations = spec.submodule_search_locations
                return alias_spec
            if fullname != _NATIVE_MODULE:
                return None
            spec = PathFinder.find_spec(fullname, path, target)
            if spec is None:
                raise ImportError(_GERT_IMPORT_ERROR) from None
            if spec.loader is not None:
                spec.loader = _NativeModuleLoader(spec.loader)
            return spec

    # Register before importing any GERT dependencies. The marker also avoids
    # installing another finder if the canonical package is reloaded.
    _finder = next(
        (finder for finder in sys.meta_path if getattr(finder, "_gert_package", None) == __name__),
        None,
    )
    _installed_finder = _finder is None
    if _installed_finder:
        _finder = _GertModuleFinder()
        sys.meta_path.insert(0, _finder)

    try:
        from fxrt.gert_backend._backend import backend
        from fxrt.gert_backend._session import CompiledModel, compile_executor
    except BaseException:
        if _installed_finder:
            sys.meta_path.remove(_finder)
        raise

    __all__ = ["backend", "compile_executor", "CompiledModel"]

    def check_schema_compatibility() -> int:
        """Raise unless torch_npu's IR export schema is the one this build reads.

        The native module enforces this on every Convert as well; calling it up front turns a
        version skew into a clear error at import or setup time rather than mid-compile.
        """
        from fxrt import _fxrt_ir  # pylint: disable=import-outside-toplevel

        from fxrt.gert_backend import _fxrt_fx_converter  # pylint: disable=import-outside-toplevel,import-self

        produced = _fxrt_ir.IR_EXPORT_SCHEMA_VERSION
        consumed = _fxrt_fx_converter.IR_EXPORT_SCHEMA_VERSION
        if produced != consumed:
            raise RuntimeError(
                f"fx_converter reads IR export schema version {consumed}, but the installed "
                f"torch_npu produces version {produced}. Rebuild fx_converter against a "
                f"matching torch_npu source tree."
            )
        return consumed

    # The finder handles modules imported later; mirror the package and all
    # dependencies already loaded while initializing it as well.
    for _name, _module in list(sys.modules.items()):
        if _name == __name__ or _name.startswith(__name__ + "."):
            sys.modules[_ALIAS_PACKAGE + _name[len(__name__):]] = _module
