"""DTensor RNG support for the NPU device.

Upstream PyTorch (the DTensor RNG tracker registry and the DTensor random
higher-order operator) lets third-party backends register:

* a device-specific RNG tracker via
  ``torch.distributed.tensor._random.register_rng_tracker``, and
* the ``run_dtensor_rng_op`` dispatch key via
  ``torch._prims.rng_prims.register_run_dtensor_rng_dispatch``

instead of hardcoding the CUDA philox assumption in core. This module wires
the NPU device into those registries.

NPU's generator state is philox-layout compatible (16 bytes: uint64 seed +
uint64 offset, offset in multiples of 4), so the offset-based machinery of
``OffsetBasedRNGTracker`` applies as-is, with one documented semantic
difference versus CUDA:

* On CUDA, the offset indexes a window into a single value stream:
  generating ``n`` values at offset ``O`` reproduces stream positions
  ``[O, O + n)`` of the base stream.
* On NPU, each offset value selects an independent philox stream, and an RNG
  kernel launch advances the offset by a fixed stride (12) regardless of how
  many values were consumed.

DTensor random ops on NPU are therefore deterministic and reproducible, and
distinct shards draw from distinct (uncorrelated) philox streams, but shard
values are NOT bitwise-equal to the corresponding slices of an unsharded
``torch.rand`` call. Rank 0's shard (start offset increment 0) does match the
prefix of the unsharded stream at the same generator state.
"""

from torch.distributed.tensor._random import OffsetBasedRNGTracker

__all__ = ["NpuRNGStateTracker", "register_npu_dtensor_rng"]


class NpuRNGStateTracker(OffsetBasedRNGTracker):
    """Offset-based DTensor RNG tracker for the NPU device.

    The NPU generator state layout (16-byte seed/offset) and the
    ``get_rng_state``/``set_rng_state`` APIs match the philox contract of
    ``OffsetBasedRNGTracker``, so the inherited offset bookkeeping applies
    unchanged. See the module docstring for the NPU stream semantics.
    """


def register_npu_dtensor_rng() -> bool:
    """Register the NPU DTensor RNG tracker and HOP dispatch key.

    Registers ``NpuRNGStateTracker`` for the ``npu`` device type and the
    device-generic ``run_dtensor_rng_op`` implementation for the
    ``PrivateUse1`` dispatch key, so DTensor random ops on NPU run through
    the traceable higher-order operator path.

    Returns ``True`` when registered, ``False`` on torch versions without
    the upstream registration APIs (in which case this is a no-op and keeps
    torch_npu importable across torch versions).
    """
    try:
        from torch._C import DispatchKey
        from torch._prims.rng_prims import register_run_dtensor_rng_dispatch
        from torch.distributed.tensor._random import register_rng_tracker
    except ImportError:
        return False

    register_rng_tracker("npu", NpuRNGStateTracker)
    register_run_dtensor_rng_dispatch(DispatchKey.PrivateUse1)
    return True
