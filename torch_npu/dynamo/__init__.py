import os
import sys
import time
import warnings

from torch._dynamo import register_backend as _register_backend
from torch._dynamo.backends.registry import _BACKENDS

from torch_npu.utils._error_code import ErrCode, pta_error
from torch_npu.utils.utils import _should_print_warning
from .trace_rule import _patch_npu_trace_rules

_global_npu_backend = {}
__all__ = []
_global_backend_name = None

NPUGRAPH_EX_BACKEND = "npugraph_ex"


class _ImportError(Exception):
    def __init__(self, pkg_name):
        super().__init__(self)
        self.err_info = (
            "\nAn error occurred when import `" + pkg_name + "` and the above is the specific error message. \n"
            "This error message was generated when import " + pkg_name + ", but thrown asynchronously here. \n"
            "Please check the error message above. \n") + pta_error(ErrCode.INTERNAL)

    def __str__(self):
        return self.err_info


class _LazyException:
    def __init__(self, e, pkg_name):
        self._info = _ImportError(pkg_name)
        self._e = e

    def __getattr__(self, name):
        raise self._info from self._e

    def __call__(self, *args, **kwargs):
        raise self._info from self._e


def _eager_npu_backend(gm, *args, **kwargs):
    return gm


def _get_global_npu_backend(name, config=None):
    global _global_npu_backend
    if name != NPUGRAPH_EX_BACKEND and name in _global_npu_backend.keys():
        return _global_npu_backend[name]
    if 'torchair' not in sys.modules:
        raise AssertionError("Could not find module torchair. "
                             "Please check if torchair is removed from sys.modules." + pta_error(ErrCode.NOT_FOUND))
    import torchair
    _global_npu_backend[name] = torchair.get_npu_backend(compiler_config=config)
    return _global_npu_backend[name]


class _LazyBackend:
    def __init__(self):
        self._exception = None
        self._allowed_list = ["__spec__", "__path__"]

    def _pta_error_code(self):
        # Use static error code here because pta_error will lazy init the torch_npu's submodule,
        # which will cause error in `for loop` of sys.modules, code like:
        # - for m in sys.modules.values():
        # -     getattr(m, name, None)
        error_msg = "\n[ERROR] {time} (PID:{pid}, Device:-1, RankID:-1) ERR00005 PTA internal error"
        return error_msg.format(
            time=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
            pid=os.getpid())


class _LazyTorchair(_LazyBackend):
    def __init__(self, pkg_name):
        self._torchair = None
        self._pkg_name = pkg_name
        super().__init__()

    def __getattr__(self, name):
        if self._exception is not None:
            return self._exception()

        if self._torchair is not None:
            return getattr(self._torchair, name)

        if name not in self._allowed_list:
            raise AttributeError(f"Try to get {self._pkg_name}'s attr `{name}` before {self._pkg_name} is initialized."
                                 + self._pta_error_code())

        try:
            from . import torchair
        except Exception as e:
            # In cpython, default import loader will suppress error when
            # find module's __spec__. So here we need to record error and
            # replay it later (when this func is invoked again).
            self._exception = _LazyException(e, self._pkg_name)
            raise

        self._torchair = torchair
        return getattr(torchair, name)


class _LazyNpuGraphEx(_LazyBackend):
    def __init__(self, pkg_name):
        self._npugraph_ex = None
        self._pkg_name = pkg_name
        super().__init__()

    def __getattr__(self, name):
        if self._exception is not None:
            return self._exception()

        if self._npugraph_ex is not None:
            return getattr(self._npugraph_ex, name)

        if name not in self._allowed_list:
            raise AttributeError(f"Try to get {self._pkg_name}'s attr `{name}` before {self._pkg_name} is initialized."
                                 + self._pta_error_code())

        try:
            from . import npugraph_ex
        except Exception as e:
            self._exception = _LazyException(e, self._pkg_name)
            raise

        self._npugraph_ex = npugraph_ex
        return getattr(npugraph_ex, name)


def _lazy_exec(*args, **kwargs):
    global _global_backend_name
    return _get_global_npu_backend(_global_backend_name)(*args, **kwargs)


def _get_default_backend(name):
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'torchair')):
        if _should_print_warning():
            warnings.warn(
                "Register eager implementation for the 'npu' backend of dynamo, "
                "as torch_npu was not compiled with torchair.")
        return _eager_npu_backend
    global _global_backend_name
    _global_backend_name = name
    sys.modules['torchair'] = _LazyTorchair('torchair')
    return _lazy_exec


def _exec(*args, **kwargs):
    import npugraph_ex
    config = npugraph_ex.CompilerConfig()
    config.mode = NPUGRAPH_EX_BACKEND
    return npugraph_ex.get_npu_backend(compiler_config=config)(*args, **kwargs)


def _get_npugraph_ex_backend():
    sys.modules[NPUGRAPH_EX_BACKEND] = _LazyNpuGraphEx(NPUGRAPH_EX_BACKEND)
    return _exec


_global_backend = _get_default_backend(name="npu")
_npugraph_ex_backend = _get_npugraph_ex_backend()


def _register_npu_backend(backend, name="npu"):
    if name in _BACKENDS.keys():
        del _BACKENDS[name]
    _register_backend(backend, name)


_register_npu_backend(_global_backend)
_register_npu_backend(_npugraph_ex_backend, NPUGRAPH_EX_BACKEND)