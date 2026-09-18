"""The batch splitting policy feeding :class:`DualBatchRunner`.

The default policy splits a batch whose real size sits far below the next
power-of-two execution gear: the main lane takes the largest power of two
below the batch, the offset lane takes the remainder.  Batches that already
land on a power of two (128, 256, ...) or whose padding ratio is below the
threshold keep single-stream execution, where no padding is wasted.

``PTA_DUAL_STREAM_PAD_RATIO`` (default 0.4) selects the minimum single-stream
padding ratio that justifies a split.
"""

import logging
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


log = logging.getLogger("torch_npu.dual_stream_graph")

# Batches below this size never pay the two-lane overhead.
MIN_SPLITTABLE_BATCH = 16


def _pad_ratio_threshold() -> float:
    raw = os.getenv("PTA_DUAL_STREAM_PAD_RATIO")
    if raw is None:
        return 0.4
    try:
        ratio = float(raw)
    except ValueError:
        log.warning("invalid PTA_DUAL_STREAM_PAD_RATIO=%r, falling back to 0.4", raw)
        return 0.4
    if not 0.0 <= ratio < 1.0:
        log.warning(
            "PTA_DUAL_STREAM_PAD_RATIO=%r out of [0, 1), falling back to 0.4", raw
        )
        return 0.4
    return ratio


def plan_split(batch_size: int) -> Optional[int]:
    """Return the main-lane size for ``batch_size``, or None to stay single-stream.

    The offset lane then receives ``batch_size - main``.  Returns None for
    batches that are too small, already on a power-of-two gear, or whose
    padding ratio does not justify the split.
    """
    if batch_size < max(2, MIN_SPLITTABLE_BATCH) or not (batch_size & (batch_size - 1)):
        return None
    split = 1 << (batch_size.bit_length() - 1)
    padded = 1 << batch_size.bit_length()
    if (padded - batch_size) / padded <= _pad_ratio_threshold():
        return None
    return split


def _first_batch_tensor(args: Sequence[Any], kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
    for value in tuple(args) + tuple(kwargs.values()):
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return value
    return None


def plan_chunks(
    *args: Any, **kwargs: Any
) -> Optional[Tuple[Tuple[Tuple[tuple, dict], Tuple[tuple, dict]], Tuple[int, int]]]:
    """Split one call's arguments into two lane groups plus their lanes.

    The first batched tensor input found among args/kwargs values drives the
    split; it is split along dim 0 with :func:`torch.split` and substituted
    back into its original position.  Returns ``((group0, group1), (0, 1))``
    or None when the policy decides the call must stay single-stream.
    """
    tensor = _first_batch_tensor(args, kwargs)
    if tensor is None:
        return None
    split = plan_split(int(tensor.shape[0]))
    if split is None:
        return None
    parts = torch.split(tensor, [split, int(tensor.shape[0]) - split], dim=0)

    def build(size_index: int) -> Tuple[tuple, dict]:
        def substitute(value: Any) -> Any:
            if value is tensor:
                return parts[size_index]
            return value

        return (
            tuple(substitute(value) for value in args),
            {key: substitute(value) for key, value in kwargs.items()},
        )

    return (build(0), build(1)), (0, 1)
