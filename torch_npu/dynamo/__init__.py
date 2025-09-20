import os
import sys
import time
import warnings
import importlib

from torch._dynamo import register_backend as _register_backend
from torch._dynamo.backends.registry import _BACKENDS
from torch.library import Library, impl

from torch_npu.utils._error_code import ErrCode, pta_error
from torch_npu.utils.utils import _should_print_warning

_global_npu_backend = None
__all__ = []


class _TorchairImportError(Exception):
    def __init__(self):
        super().__init__(self)
        self.err_info = (
            "\nAn error occured when import `torchair` and the above is the specific error message. \n"
            "This error message was generated when import torchair, but throwed asynchronously here. \n"
            "Please check the error message above. \n") + pta_error(ErrCode.INTERNAL)

    def __str__(self):
        return self.err_info


class _LazyException:
    def __init__(self, e):
        self._info = _TorchairImportError()
        self._e = e

    def __getattr__(self, name):
        raise self._info from self._e

    def __call__(self, *args, **kwargs):
        raise self._info from self._e


def _eager_npu_backend(gm, *args, **kwargs):
    return gm


def _get_global_npu_backend():
    global _global_npu_backend
    if _global_npu_backend is not None:
        return _global_npu_backend
    if 'torchair' not in sys.modules:
        raise AssertionError("Could not find module torchair. "
                             "Please check if torchair is removed from sys.modules." + pta_error(ErrCode.NOT_FOUND))
    import torchair
    _global_npu_backend = torchair.get_npu_backend()
    return _global_npu_backend


class _LazyTorchair:
    def __init__(self):
        self._torchair = None
        self._exception = None
        self._allowed_list = ["__spec__", "__path__"]

    def __getattr__(self, name):
        if self._exception is not None:
            return self._exception()

        if self._torchair is not None:
            return getattr(self._torchair, name)

        if name not in self._allowed_list:
            raise AttributeError(f"Try to get torchair's attr `{name}` before torchair is initialized."
                                    + self._pta_error_code())

        try:
            from . import torchair
        except Exception as e:
            # In cpython, default import loader will suppress error when
            # find module's __spec__. So here we need to record error and
            # replay it later (when this func is invoked again).
            self._exception = _LazyException(e)
            raise

        self._torchair = torchair
        return getattr(torchair, name)

    def _pta_error_code(self):
        # Use static error code here because pta_error will lazy init the torch_npu's submodule,
        # which will cause error in `for loop` of sys.modules, code like:
        # - for m in sys.modules.values():
        # -     getattr(m, name, None)
        error_msg = "\n[ERROR] {time} (PID:{pid}, Device:-1, RankID:-1) ERR00005 PTA internal error"
        return error_msg.format(
            time=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
            pid=os.getpid())


def _get_default_backend():
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'torchair')):
        if _should_print_warning():
            warnings.warn(
                "Register eager implementation for the 'npu' backend of dynamo, "
                "as torch_npu was not compiled with torchair.")
        return _eager_npu_backend

    def _lazy_exec(*args, **kwargs):
        return _get_global_npu_backend()(*args, **kwargs)

    sys.modules['torchair'] = _LazyTorchair()
    return _lazy_exec


_global_backend = _get_default_backend()


def _register_npu_backend(backend):
    if 'npu' in _BACKENDS.keys():
        del _BACKENDS['npu']
    _register_backend(backend, 'npu')


_register_npu_backend(_global_backend)
