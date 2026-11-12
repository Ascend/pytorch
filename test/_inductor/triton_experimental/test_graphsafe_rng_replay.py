# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Regression tests for graphsafe RNG replay in triton_experimental.

These tests validate that random operators captured by the NPU Inductor path
receive explicit, replayable generator state. The direct HOP test covers the
NPU graphsafe RNG registration, and the checkpoint/dropout test covers the
AOTAutograd + Inductor GeneratorState path used by training recomputation.
"""

import os
from unittest.mock import patch

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
import torch_npu.utils._inductor  # noqa: F401

needs_npu = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an NPU device"
)


@needs_npu
def test_graphsafe_run_with_rng_state_replays_and_restores_npu_state():
    from torch._functorch._aot_autograd.utils import supports_graphsafe_rng
    from torch._prims.rng_prims import graphsafe_run_with_rng_state

    assert supports_graphsafe_rng(torch.device("npu"))

    device = torch.device("npu:0")
    x = torch.arange(1, 65, device=device, dtype=torch.float32).reshape(8, 8)

    def dropout(t):
        return F.dropout(t, p=0.4, training=True)

    torch.manual_seed(58)
    torch.npu.manual_seed(58)
    generator = torch.npu._get_generator(device)
    rng_state = generator.clone_state()
    expected_first = dropout(x)
    expected_second = dropout(x)

    torch.manual_seed(999)
    torch.npu.manual_seed(999)
    ambient_state = generator.clone_state()
    ambient_expected = dropout(x)

    generator.graphsafe_set_state(ambient_state)
    actual_first = graphsafe_run_with_rng_state(dropout, x, rng_state=rng_state)
    actual_second = graphsafe_run_with_rng_state(dropout, x, rng_state=rng_state)
    ambient_actual = dropout(x)

    torch.testing.assert_close(actual_first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(actual_second, expected_second, rtol=0, atol=0)
    torch.testing.assert_close(ambient_actual, ambient_expected, rtol=0, atol=0)


@needs_npu
def test_philox_seed_offset_helpers_are_npu_aware():
    from torch._decomp import decompositions_for_rng
    from torch._prims import rng_prims

    assert getattr(rng_prims.CUDARngStateHelper, "_npu_patched", False)
    assert getattr(decompositions_for_rng.throw_on_non_cuda, "_npu_patched", False)
    assert decompositions_for_rng.throw_on_non_cuda(torch.device("npu")) is None

    assert int(rng_prims.philox_rand_offset((2, 3, 4)).item()) == 24
    assert int(rng_prims.philox_rand_offset((7,)).item()) == 7


def _compiled_rand_pair(x):
    def fn(t):
        first = torch.rand_like(t)
        second = torch.rand_like(t)
        return first, second

    return torch.compile(
        fn,
        backend="inductor",
        fullgraph=True,
        options={"npu_backend": "triton_experimental"},
    )(x)


@needs_npu
def test_functionalized_seed_offset_replays_in_triton_experimental():
    x = torch.arange(1, 65, device="npu", dtype=torch.float32).reshape(8, 8)

    with patch.object(torch._functorch.config, "functionalize_rng_ops", True):
        torch._dynamo.reset()

        torch.manual_seed(2026)
        torch.npu.manual_seed(2026)
        first_a, first_b = _compiled_rand_pair(x)
        torch.npu.synchronize()

        torch.manual_seed(2026)
        torch.npu.manual_seed(2026)
        replay_a, replay_b = _compiled_rand_pair(x)
        torch.npu.synchronize()

        torch.manual_seed(2027)
        torch.npu.manual_seed(2027)
        different_a, _ = _compiled_rand_pair(x)
        torch.npu.synchronize()

    torch.testing.assert_close(first_a.cpu(), replay_a.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(first_b.cpu(), replay_b.cpu(), rtol=0, atol=0)
    assert not torch.equal(first_a.cpu(), first_b.cpu())
    assert not torch.equal(first_a.cpu(), different_a.cpu())


def _checkpoint_dropout(x):
    def inner(t):
        return F.dropout(t, p=0.4, training=True)

    return checkpoint(inner, x, use_reentrant=False, preserve_rng_state=True)


def _run_compiled_checkpoint_dropout(seed, shape=(8, 8)):
    torch._dynamo.reset()
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)

    x = torch.arange(
        1, 1 + shape[0] * shape[1], device="npu", dtype=torch.float32
    ).reshape(*shape).requires_grad_(True)
    opt = torch.compile(
        _checkpoint_dropout,
        backend="inductor",
        options={"npu_backend": "triton_experimental"},
    )
    out = opt(x)
    out.sum().backward()
    torch.npu.synchronize()

    scale = 1.0 / (1.0 - 0.4)
    mask_from_output = torch.round(
        out.detach().cpu() / x.detach().cpu() / scale
    ).to(torch.int32)
    mask_from_grad = torch.round(x.grad.detach().cpu() / scale).to(torch.int32)
    assert torch.equal(mask_from_output, mask_from_grad)
    return out.detach().cpu(), x.grad.detach().cpu()


@needs_npu
def test_checkpoint_dropout_preserves_rng_state_in_triton_experimental():
    out1, grad1 = _run_compiled_checkpoint_dropout(2026, shape=(8, 8))
    out2, grad2 = _run_compiled_checkpoint_dropout(2026, shape=(8, 8))
    out3, grad3 = _run_compiled_checkpoint_dropout(2026, shape=(4, 16))
    out4, grad4 = _run_compiled_checkpoint_dropout(2026, shape=(4, 16))

    torch.testing.assert_close(out1, out2, rtol=0, atol=0)
    torch.testing.assert_close(grad1, grad2, rtol=0, atol=0)
    torch.testing.assert_close(out3, out4, rtol=0, atol=0)
    torch.testing.assert_close(grad3, grad4, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
