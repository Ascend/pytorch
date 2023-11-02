import os
import sys
import warnings

from torch._dynamo import register_backend as _register_backend
from torch._dynamo.backends.registry import _BACKENDS
from torch.library import Library, impl


__all__ = []


def _eager_npu_backend(gm, *args, **kwargs):
    return gm


def _get_default_backend():
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'torchair')):
        warnings.warn(
            "Register eager implementation for the 'npu' backend of dynamo, "
            "as torch_npu was not compiled with torchair.")
        return _eager_npu_backend

    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from . import torchair
        return torchair.get_npu_backend()
    finally:
        del sys.path[0]


_global_backend = _get_default_backend()


def _register_npu_backend(backend):
    if 'npu' in _BACKENDS.keys():
        del _BACKENDS['npu']
    _register_backend(backend, 'npu')


_register_npu_backend(_global_backend)
