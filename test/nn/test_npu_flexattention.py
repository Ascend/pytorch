# Copyright (c) 2026, Huawei Technologies Co., Ltd
"""Verify flex_attention runs in eager mode on NPU (pre-2.11 torch).

On torch < 2.11 the community does not yet support NPU autocast for
flex_attention, so this test only covers eager execution and device
validation.  Autocast coverage lives in the master/v2.11+ test file.
"""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import flex_attention


def _noop_score_mod(score, b, h, m, n):
    return score


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


if __name__ == "__main__":
    test_eager_flex_attention()
    print("PASS: NPU FlexAttention eager execution.")
