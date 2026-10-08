"""NPU linalg_qr fallback output layout regression tests (issue #5072).

torch_npu eager linalg_qr (aclnn) returns row-major Q/R while the ATen meta
kernel declares the LAPACK column-major convention. Inductor used to build
fallback output layouts from that meta, so compiled consumers of QR outputs
read them with transposed strides (svd_lowrank / pca_lowrank spectrum
collapse). These tests compile exactly those consumption patterns and assert
element-wise equality with eager.
"""

import pytest
import torch

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

needs_npu = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an NPU device"
)


@needs_npu
def test_qr_output_consumed_by_mm():
    torch.manual_seed(0)
    A = torch.randn(32, 16, device="npu")
    R = torch.randn(16, 8, device="npu")

    def chain(a, r):
        q = torch.linalg.qr(a @ r)[0]
        return q.mH @ a

    assert (torch.compile(chain)(A, R) - chain(A, R)).abs().max().item() < 1e-4


@needs_npu
def test_lowrank_chain_with_fixed_random():
    torch.manual_seed(0)
    A = torch.randn(32, 16, device="npu")
    R = torch.randn(16, 8, device="npu")

    def chain(a, r, niter=2):
        q = torch.linalg.qr(a @ r)[0]
        for _ in range(niter):
            q = torch.linalg.qr(a.mH @ q)[0]
            q = torch.linalg.qr(a @ q)[0]
        b = q.mH @ a
        u, s, vh = torch.linalg.svd(b, full_matrices=False)
        return q @ u, s, vh

    e_u, e_s, e_vh = chain(A, R)
    c_u, c_s, c_vh = torch.compile(chain)(A, R)
    for compiled, eager in ((c_u, e_u), (c_s, e_s), (c_vh, e_vh)):
        assert (compiled - eager).abs().max().item() < 1e-4
    # svd_lowrank is a randomized rank-q approximation; both eager and
    # compiled carry the same O(0.5) spectral budget on random inputs.
    # The assert above is what pins compiled to eager.
    ref = torch.linalg.svdvals(A.double())[: R.shape[-1]]
    assert (c_s - ref.float()).abs().max().item() < 0.6


@needs_npu
def test_batched_qr_projector():
    torch.manual_seed(0)
    X = torch.randn(3, 10, 8, device="npu")

    def proj(t):
        q = torch.linalg.qr(t)[0]
        return q @ q.mT

    assert (torch.compile(proj)(X) - proj(X)).abs().max().item() < 1e-4


@needs_npu
def test_qr_complete_and_r_modes():
    torch.manual_seed(0)
    A = torch.randn(10, 6, device="npu")

    qc, rc = torch.compile(lambda t: torch.linalg.qr(t, mode="complete"))(A)
    ec, er = torch.linalg.qr(A, mode="complete")
    assert (qc - ec).abs().max().item() < 1e-4
    assert (rc - er).abs().max().item() < 1e-4

    _, rr = torch.compile(lambda t: torch.linalg.qr(t, mode="r"))(A)
    _, er2 = torch.linalg.qr(A, mode="r")
    assert (rr - er2).abs().max().item() < 1e-4


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
