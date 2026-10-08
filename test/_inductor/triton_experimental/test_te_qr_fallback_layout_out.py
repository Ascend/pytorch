"""`.out` overload guard for the QR fallback-layout fix (#5072, !47273).

Two guards:

1. ``test_qr_out_compiled_matches_eager`` -- the compile-able `.out` path:
   with a contiguous caller-allocated output buffer the compiled graph must
   write into the *supplied* tensors (identity + strides preserved) and
   produce values equal to eager.  Without the meta fix this fails
   element-wise (the fallback layout mismatch is read as a transposed
   buffer); with the fix it passes bitwise-close -- this is the regression
   guard for the ``aten.linalg_qr.out`` half of the patch.

2. ``test_qr_out_noncontiguous_views_rejected_by_dynamo`` -- pins the
   reviewer-suggested shape (transposed-view outputs): Dynamo rejects
   non-contiguous ``out=`` tensors at tracing time (graph break gb0241),
   before Inductor runs, so that case cannot serve as a compiled-path
   guard today.  If upstream Dynamo ever learns to trace it, this test
   fails and flags guard 1's buffer to be upgraded.
"""

import pytest
import torch
import torch._dynamo

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

needs_npu = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an NPU device"
)


def _qr_and_consume(a, q_out, r_out):
    q, r = torch.linalg.qr(a, out=(q_out, r_out))
    return q, r, q @ r


@needs_npu
def test_qr_out_compiled_matches_eager():
    torch.manual_seed(0)
    a = torch.randn(8, 4, device="npu")

    q_eager, r_eager = torch.empty((8, 4), device="npu"), torch.empty(
        (4, 4), device="npu"
    )
    eager = _qr_and_consume(a, q_eager, r_eager)

    q_compiled, r_compiled = torch.empty((8, 4), device="npu"), torch.empty(
        (4, 4), device="npu"
    )
    compiled = torch.compile(_qr_and_consume, fullgraph=True)(a, q_compiled, r_compiled)

    # The out overload must write to, and return, the supplied buffers.
    assert compiled[0].data_ptr() == q_compiled.data_ptr()
    assert compiled[1].data_ptr() == r_compiled.data_ptr()
    assert compiled[0].stride() == q_compiled.stride()
    assert compiled[1].stride() == r_compiled.stride()

    for actual, expected in zip(compiled, eager):
        torch.testing.assert_close(actual, expected)


@needs_npu
def test_qr_out_noncontiguous_views_rejected_by_dynamo():
    # Reviewer-suggested shape: QR results routed into transposed views of
    # row-major storage (Q is (8, 4) strided (1, 8)).  Inductor's fallback
    # never sees this -- Dynamo refuses to trace non-contiguous out= tensors.
    torch.manual_seed(0)
    a = torch.randn(8, 4, device="npu")

    def make_outputs():
        q_storage = torch.empty((4, 8), device="npu")
        r_storage = torch.empty((4, 4), device="npu")
        return q_storage.mT, r_storage.mT

    q_compiled, r_compiled = make_outputs()
    with pytest.raises(torch._dynamo.exc.Unsupported, match="non-contiguous"):
        torch.compile(_qr_and_consume, fullgraph=True)(a, q_compiled, r_compiled)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
