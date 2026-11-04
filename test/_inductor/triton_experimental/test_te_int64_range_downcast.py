"""TE (triton_experimental) int64 range-aware downcast regression tests (#4947).

``torch.masked.{amax,amin,argmax,argmin}`` decompose (correctly, upstream) into
``where(mask, x, fill=<dtype identity>)`` + a real reduction. The int64 identity
is ``torch.iinfo(torch.int64)`` ``MIN``/``MAX``, which is out of int32 range. The
TE backend used to demote every int64 to int32 unconditionally (a perf heuristic
for the AIV core), so those constants were rendered as
``tl.full([1], -9223372036854775808, tl.int32)`` and wrapped on materialization:

* ``INT64_MIN -> 0``  -> a fully-masked amax row returns 0 (not INT64_MIN), and a
  0-fill beats real negatives, corrupting amax/argmax on negative rows;
* ``INT64_MAX -> -1`` -> a fully-masked amin row returns -1 (not INT64_MAX).

The same mechanism also downcast ``*i64`` data pointers to ``*i32``, truncating any
wide int64 payload (|x| >= 2^31).

The fix makes the demotion range-aware: an int64 constant is kept native
``tl.int64`` when it falls outside ``[-2^31, 2^31-1]``, and a kernel emitting such
a wide constant skips its ``*i64`` pointer downcast so the result is not
re-truncated through the ``*i32`` output pointer. These tests assert both the
end-to-end numbers (vs eager) and the generated kernel shape (native tl.int64
fill, no i32 wrapping). Default backend is untouched.
"""

import os

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import pytest
import torch
from torch._inductor import config
from torch._inductor.utils import run_and_get_code

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

INT64_MIN = torch.iinfo(torch.int64).min  # -9223372036854775808
INT64_MAX = torch.iinfo(torch.int64).max  # 9223372036854775807

requires_npu = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an NPU device"
)


def _run_te(fn, *inputs):
    """Compile ``fn`` under TE strict mode, run it, and return (result, source).

    ``source`` is the concatenation of every generated Triton kernel body, so a
    test can pin the exact emitted constant type / pointer signature in addition
    to the numerics.
    """
    compiled = torch.compile(
        fn,
        backend="inductor",
        fullgraph=True,
        options={"npu_backend": "triton_experimental"},
    )
    with (
        config.patch("force_disable_caches", True),
        config.patch("implicit_fallbacks", False),
    ):
        result, codes = run_and_get_code(compiled, *inputs)
    # Loud failure if a stale TORCHINDUCTOR_NPU_BACKEND import order silently
    # selected another backend (mirror of the sibling registrar-gap test).
    assert (
        torch_npu.utils._dynamo._InductorNpuRegistry._loaded_backend
        == "triton_experimental"
    ), "triton_experimental backend did not take effect for this compile"
    return result, "\n".join(codes)


def _masked(op_name):
    op = getattr(torch.masked, op_name)
    return lambda a, mask: op(a, 1, keepdim=True, mask=mask)


# --- masked int64 reduction identities ------------------------------------


@requires_npu
def test_masked_amax_int64_fullmask_identity():
    """P4D-1: a fully-masked row must return INT64_MIN, not 0."""
    a = torch.tensor([[1, 9, 3, 2, 4], [5, 6, 7, 8, 9]], dtype=torch.int64)
    m = torch.tensor(
        [[True, True, True, True, True], [False, False, False, False, False]]
    )
    a, m = a.npu(), m.npu()
    eager = torch.masked.amax(a, 1, keepdim=True, mask=m)
    assert eager.tolist() == [[9], [int(INT64_MIN)]]

    got, source = _run_te(_masked("amax"), a, m)
    assert torch.equal(got, eager), f"got {got.tolist()} want {eager.tolist()}"
    # the identity is rendered native int64, never wrapped through int32.
    assert f"{int(INT64_MIN)}, tl.int64" in source
    assert f"{int(INT64_MIN)}, tl.int32" not in source


@requires_npu
def test_masked_amin_int64_fullmask_identity():
    """P4D-1: a fully-masked row must return INT64_MAX, not -1."""
    a = torch.tensor([[1, 9, 3, 2, 4], [5, 6, 7, 8, 9]], dtype=torch.int64)
    m = torch.tensor(
        [[True, True, True, True, True], [False, False, False, False, False]]
    )
    a, m = a.npu(), m.npu()
    eager = torch.masked.amin(a, 1, keepdim=True, mask=m)
    assert eager.tolist() == [[1], [int(INT64_MAX)]]

    got, source = _run_te(_masked("amin"), a, m)
    assert torch.equal(got, eager), f"got {got.tolist()} want {eager.tolist()}"
    assert f"{int(INT64_MAX)}, tl.int64" in source
    assert f"{int(INT64_MAX)}, tl.int32" not in source


@requires_npu
def test_masked_amax_int64_negative_identity_dominates():
    """P4D-2: on a negative row the 0-wrapped fill used to beat the real max."""
    a = torch.tensor([[-5, -2, -3, -1, -4], [1, 2, 3, 4, 5]], dtype=torch.int64)
    m = torch.tensor(
        [[True, True, False, True, True], [False, False, False, False, False]]
    )
    a, m = a.npu(), m.npu()
    eager = torch.masked.amax(a, 1, keepdim=True, mask=m)
    # row0: max of {-5,-2,-1,-4} = -1 ; row1: fully masked -> INT64_MIN
    assert eager.tolist() == [[-1], [int(INT64_MIN)]]
    got, _ = _run_te(_masked("amax"), a, m)
    assert torch.equal(got, eager), f"got {got.tolist()} want {eager.tolist()}"


@requires_npu
def test_masked_argmax_int64_negative_values():
    """P4D-2: argmax index off-by-one when the 0-fill beat the real max."""
    a = torch.tensor([[-5, -2, -3, -1, -4]], dtype=torch.int64)
    m = torch.tensor([[True, True, False, True, True]])
    a, m = a.npu(), m.npu()
    eager = torch.masked.argmax(a, 1, keepdim=True, mask=m)
    # largest among masked-in positions (idx 0,1,3,4) is -1 at index 3
    assert eager.tolist() == [[3]]
    got, _ = _run_te(_masked("argmax"), a, m)
    assert torch.equal(got, eager), f"got {got.tolist()} want {eager.tolist()}"


# --- wide int64 payload (second leg of the same mechanism) -----------------


@requires_npu
def test_masked_amax_int64_wide_payload():
    """P4D-6: |x| >= 2^31 payload used to truncate via the *i32 pointer."""
    payload = 2**40
    a = torch.tensor([[payload + 1, payload + 2]], dtype=torch.int64)
    m = torch.tensor([[True, True]])
    a, m = a.npu(), m.npu()
    eager = torch.masked.amax(a, 1, keepdim=True, mask=m)
    assert eager.tolist() == [[payload + 2]]
    got, _ = _run_te(_masked("amax"), a, m)
    assert torch.equal(got, eager), f"got {got.tolist()} want {eager.tolist()}"


# --- no-regression controls (int32 identity in-range; float path) ----------


@requires_npu
def test_control_masked_amax_int32_no_regression():
    """INT32_MIN == -2^31 is in range -> normal demotion path, exact result."""
    a = torch.tensor([[1, 9, 3], [5, 6, 7]], dtype=torch.int32)
    m = torch.tensor([[True, True, True], [False, False, False]])
    a, m = a.npu(), m.npu()
    eager = torch.masked.amax(a, 1, keepdim=True, mask=m)
    assert eager.tolist() == [[9], [int(torch.iinfo(torch.int32).min)]]
    got, _ = _run_te(_masked("amax"), a, m)
    assert torch.equal(got, eager)


@requires_npu
def test_control_masked_amax_float32_no_regression():
    a = torch.tensor([[1.0, 9.0, 3.0], [5.0, 6.0, 7.0]], dtype=torch.float32)
    m = torch.tensor([[True, True, True], [False, False, False]])
    a, m = a.npu(), m.npu()
    eager = torch.masked.amax(a, 1, keepdim=True, mask=m)
    got, _ = _run_te(_masked("amax"), a, m)
    torch.testing.assert_close(got, eager)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
