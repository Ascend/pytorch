"""NPU implementations for the jagged NestedTensor packing ops.

``torch.nested.to_padded_tensor`` on a jagged NestedTensor, and the reverse
conversion done by ``torch.nested.nested_tensor_from_jagged``, are built on
``aten::_jagged_to_padded_dense_forward`` and
``aten::_padded_dense_to_jagged_forward``.  Neither op has an NPU kernel, so
both used to fall back to the CPU (device-to-host copy, CPU compute, copy
back, forced synchronisation).

Both conversions are pure data movement, so instead of a dedicated kernel they
are expressed with operators that already run on NPU: source row indices built
from the offsets with ``arange``, a gather along the packed dimension, and a
masked read/write for the padded entries.
"""

import torch


_JAGGED_TO_PADDED = "_jagged_to_padded_dense_forward"
_PADDED_TO_JAGGED = "_padded_dense_to_jagged_forward"

_registered = False
_library = None


def _dense_mask(lengths, max_len):
    """``(batch, max_len)`` mask selecting the real, non-padding entries."""
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _padding_value(padding_value, dtype):
    """Convert the ``float`` padding sentinel into a value ``dtype`` can hold.

    The schema takes ``padding_value`` as a ``float``, so ``int64`` limits (and
    the infinities upstream uses as "pad with the dtype max/min" sentinels)
    overflow the double -> int conversion; the CPU kernel clamps them instead.
    """
    if dtype.is_floating_point or dtype.is_complex or dtype is torch.bool:
        return padding_value

    limits = torch.iinfo(dtype)
    if padding_value >= limits.max:
        return limits.max
    if padding_value <= limits.min:
        return limits.min
    return int(padding_value)


def _check_offsets(offsets, op_name):
    if offsets.dim() != 1:
        raise RuntimeError(
            f"{op_name}(): expected 1D offsets, but got offsets.dim() == {offsets.dim()}"
        )
    return offsets.to(dtype=torch.long)


def jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
    if len(offsets) != 1 or len(max_lengths) != 1:
        raise NotImplementedError(
            "_jagged_to_padded_dense_forward(): only a single jagged dim is supported, "
            f"but got {len(offsets)} offsets and {len(max_lengths)} max_lengths"
        )

    offsets = _check_offsets(offsets[0], _JAGGED_TO_PADDED).to(device=values.device)
    max_len = int(max_lengths[0])
    lengths = offsets[1:] - offsets[:-1]
    batch, total = lengths.numel(), values.shape[0]
    tail = values.shape[1:]
    padding = _padding_value(padding_value, values.dtype)

    if batch == 0 or max_len == 0 or total == 0:
        return values.new_full((batch, max_len, *tail), padding)

    # Padding position (i, j) is copied from the ``values`` row ``offsets[i] + j``.
    # Clamping only keeps the gather in bounds: positions at or past the end of a
    # sequence are masked out below, which handles short sequences, sequences
    # longer than ``max_len`` (truncated, like the CPU kernel) and values past
    # the final offset (ignored) alike.
    src = offsets[:-1].unsqueeze(1) + torch.arange(
        max_len, device=values.device, dtype=torch.long
    )
    mask = src < offsets[1:].unsqueeze(1)

    out = values.index_select(0, src.clamp_(0, total - 1).reshape(-1))
    out = out.view(batch, max_len, *tail)
    return out.masked_fill(~mask.view(batch, max_len, *([1] * len(tail))), padding)


def padded_dense_to_jagged(dense, offsets, total_L=None):
    if len(offsets) != 1:
        raise NotImplementedError(
            "_padded_dense_to_jagged_forward(): only a single jagged dim is supported, "
            f"but got {len(offsets)} offsets"
        )
    if dense.dim() < 2:
        raise RuntimeError(
            "_padded_dense_to_jagged_forward(): expected padded dim >= 2, "
            f"but padded.dim() == {dense.dim()}"
        )

    offsets = _check_offsets(offsets[0], _PADDED_TO_JAGGED).to(device=dense.device)
    lengths = offsets[1:] - offsets[:-1]
    batch, max_len = dense.shape[0], dense.shape[1]
    tail = dense.shape[2:]

    if lengths.numel() > batch:
        raise RuntimeError(
            "_padded_dense_to_jagged_forward(): padded batch size "
            f"{batch} does not match offsets ({lengths.numel()})"
        )

    if total_L is not None:
        if int(lengths.clamp(max=max_len).sum()) != int(total_L):
            raise RuntimeError(
                "_padded_dense_to_jagged_forward(): final offset should match total_L value"
            )
    elif lengths.numel() and int(lengths.max()) > max_len:
        raise RuntimeError(
            "_padded_dense_to_jagged_forward(): found batch item of length "
            f"{int(lengths.max())} when max length specified by padded input is {max_len}"
        )

    if lengths.numel() == 0 or max_len == 0:
        return dense.new_zeros((0, *tail))

    flat = dense[: lengths.numel()].reshape(lengths.numel() * max_len, *tail)
    return flat[_dense_mask(lengths, max_len).reshape(-1)]


def _jagged_to_padded_dense_kernel(values, offsets, max_lengths, padding_value=0.0):
    # Registered as the PrivateUse1 kernel.  The implementation is looked up at
    # call time so tests can observe that this kernel actually ran on NPU.
    return jagged_to_padded_dense(values, offsets, max_lengths, padding_value)


def _padded_dense_to_jagged_kernel(dense, offsets, total_L=None):
    return padded_dense_to_jagged(dense, offsets, total_L)


def _has_npu_kernel(op_name):
    # Ask for a registered kernel only: the torch_npu CPU fallback is a
    # fallback function, not a kernel, and must not be mistaken for support.
    query = getattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", None)
    if query is None:
        return False
    return query(f"aten::{op_name}", "PrivateUse1")


def register_jagged_ops():
    """Register the jagged NestedTensor conversions for the NPU backend.

    Registrations are skipped when the running PyTorch does not define the
    operators, or when a native NPU kernel already exists (the composed
    implementation is only a stand-in for the missing kernel).
    """
    global _registered, _library
    if _registered:
        return

    op_names = [name for name in (_JAGGED_TO_PADDED, _PADDED_TO_JAGGED) if hasattr(torch.ops.aten, name)]
    op_names = [name for name in op_names if not _has_npu_kernel(name)]
    if not op_names:
        _registered = True
        return

    kernels = {
        _JAGGED_TO_PADDED: _jagged_to_padded_dense_kernel,
        _PADDED_TO_JAGGED: _padded_dense_to_jagged_kernel,
    }
    _library = torch.library.Library("aten", "IMPL", "PrivateUse1")
    for name in op_names:
        _library.impl(name, kernels[name], "PrivateUse1")
    _registered = True


register_jagged_ops()
