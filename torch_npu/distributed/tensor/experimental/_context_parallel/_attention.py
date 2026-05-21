import contextlib

import torch.nn.functional as F
import torch.distributed.tensor.experimental._attention as _native

from ._npu_attention import (
    npu_disable_cp_dtensor_dispatcher,
    npu_enable_cp_dtensor_dispatcher,
)


_native_enable_cp_dispatcher = _native._enable_cp_dispatcher


@contextlib.contextmanager
def _npu_enable_cp_dispatcher():
    with _native_enable_cp_dispatcher():
        npu_enable_cp_dtensor_dispatcher()
        try:
            yield
        finally:
            npu_disable_cp_dtensor_dispatcher()


# Replace dispatcher registration in the native 2.7.1 CP module.
_native._enable_cp_dispatcher = _npu_enable_cp_dispatcher
