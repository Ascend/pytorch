"""TE backend: aten.matmul_backward decomposition gap.

The default backend registers ``aten.matmul_backward`` in
``_register_triton_decompositions``; the TE registrar omitted it, so TE strict
mode (``implicit_fallbacks=False``) raised ``MissingOperatorWithoutDecomp`` on
compiled backward. These tests check TE backward compiles and gradients match
eager. Fixes #4930.
"""

import os

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import pytest
import torch
from torch._inductor import config
from torch._inductor.decomposition import decompositions

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

aten = torch.ops.aten

requires_npu = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an NPU device"
)


def _grads_match_eager(fn, *inputs, rtol=1e-2, atol=1e-3):
    """Compile ``fn`` under TE strict mode and check backward grads match eager."""
    eager_inputs = [t.detach().clone().requires_grad_(True) for t in inputs]
    fn(*eager_inputs).sum().backward()
    eager_grads = [t.grad.detach().clone() for t in eager_inputs]

    compiled = torch.compile(
        fn,
        backend="inductor",
        fullgraph=True,
        options={"npu_backend": "triton_experimental"},
    )
    compiled_inputs = [t.detach().clone().requires_grad_(True) for t in inputs]
    try:
        with (
            config.patch("force_disable_caches", True),
            config.patch("implicit_fallbacks", False),
        ):
            torch._dynamo.reset()
            # Fail loudly if the TE backend was not actually selected.
            assert (
                torch_npu.utils._dynamo._InductorNpuRegistry._loaded_backend
                == "triton_experimental"
            ), "triton_experimental backend did not take effect for this compile"
            compiled(*compiled_inputs).sum().backward()
    finally:
        torch._dynamo.reset()

    for name, eager_grad, compiled_grad in zip(
        ("A", "B", "C"),
        eager_grads,
        [t.grad for t in compiled_inputs],
    ):
        torch.testing.assert_close(
            compiled_grad,
            eager_grad,
            rtol=rtol,
            atol=atol,
            msg="grad mismatch on input " + name,
        )
    return [t.grad for t in compiled_inputs]


@requires_npu
def test_te_matmul_backward_registered():
    """After TE activation the decomposition table carries matmul_backward."""
    def fn(a, b):
        return a @ b

    _grads_match_eager(
        fn,
        torch.randn(4, 5, device="npu"),
        torch.randn(5, 6, device="npu"),
    )
    assert aten.matmul_backward.default in decompositions


@requires_npu
def test_te_matmul_nd_backward():
    """Batched matmul (3d) backward also compiles under TE."""
    def fn(a, b):
        return a @ b

    _grads_match_eager(
        fn,
        torch.randn(2, 4, 5, device="npu"),
        torch.randn(2, 5, 6, device="npu"),
    )


@requires_npu
def test_te_matrix_power_backward():
    """linalg.matrix_power(p=3) grad exercises aten.matmul_backward (nightly case)."""
    def fn(a):
        return torch.linalg.matrix_power(a, 3)

    _grads_match_eager(fn, torch.randn(2, 4, 4, device="npu"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
