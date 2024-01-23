import torch_npu

__all__ = ["Reducer", "_register_comm_hook", "_register_builtin_comm_hook",
     "_compute_bucket_assignment_by_size", "_verify_params_across_processes", "_broadcast_coalesced"]


def is_available():
    """
    Returns ``True`` if the distributed package is available. Otherwise,
    ``torch.distributed`` does not expose any other APIs. Currently,
    ``torch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """
    return hasattr(torch_npu._C, "_c10d_npu_init")


if is_available() and not torch_npu._C._c10d_npu_init():
    raise RuntimeError("Failed to initialize torch_npu.distributed")


from torch_npu._C._distributed_c10d import (
    Reducer,
    _register_comm_hook,
    _register_builtin_comm_hook,
    _compute_bucket_assignment_by_size,
    _verify_params_across_processes,
    _broadcast_coalesced
)

from .distributed_c10d import batch_isend_irecv
