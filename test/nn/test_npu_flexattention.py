# Copyright (c) 2026, Huawei Technologies Co., Ltd
"""Verify flex_attention actually runs in eager and inductor modes on NPU."""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import flex_attention


def _noop_score_mod(score, b, h, m, n):
    return score


# ── test 1: eager mode ──────────────────────────────────────────────────


def test_eager_flex_attention():
    import torch_npu  # noqa: F401
    torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

    B, H, S, D = 2, 4, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)

    output = flex_attention(q, k, v, score_mod=_noop_score_mod)
    assert output.shape == (B, H, S, D), f"eager: unexpected shape {output.shape}"
    print("  eager flex_attention OK")


# ── test 2: autocast ────────────────────────────────────────────────────


def test_npu_flex_attention_autocast():
    import torch_npu  # noqa: F401
    torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
    torch.manual_seed(0)

    batch_size = 2
    num_heads = 4
    sequence_length = 16
    head_dim = 32

    shape = (batch_size, num_heads, sequence_length, head_dim)
    query = torch.randn(shape, device="npu", dtype=torch.float32)
    key = torch.randn(shape, device="npu", dtype=torch.float32)
    value = torch.randn(shape, device="npu", dtype=torch.float32)

    # Keep an FP32 result as a numerical reference.
    expected = flex_attention(query, key, value)
    assert expected.device.type == "npu"
    assert expected.dtype == torch.float32

    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("npu")
        actual = flex_attention(query, key, value)

    assert actual.device.type == "npu"
    assert actual.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )
    print("  autocast flex_attention OK")


if __name__ == "__main__":
    test_eager_flex_attention()
    test_npu_flex_attention_autocast()
    print("PASS: NPU FlexAttention eager + inductor execution.")
