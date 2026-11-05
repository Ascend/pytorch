# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Owner(s): ["module: inductor"]
"""triton_experimental: FakeTensor dispatch cache on/off FX metadata + crosscheck.

Wraps ``compile_fx`` so the test can collect each FX node's example_value/val
metadata (string ``backend="inductor"`` does not expose the GraphModule), then
compares CACHE off vs on(+crosscheck).
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._inductor.compile_fx import compile_fx
from torch._subclasses.fake_tensor import (
    extract_tensor_metadata,
    FakeTensorMode,
)
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

requires_npu = unittest.skipUnless(torch.npu.is_available(), "requires an NPU")


def _collect_fx_tensor_meta(gm: torch.fx.GraphModule) -> dict[str, object]:
    out: dict[str, object] = {}
    for node in gm.graph.nodes:
        val = node.meta.get("example_value", node.meta.get("val"))
        if isinstance(val, torch.Tensor):
            out[node.name] = extract_tensor_metadata(val)
        elif isinstance(val, (tuple, list)):
            metas = []
            for item in val:
                if isinstance(item, torch.Tensor):
                    metas.append(extract_tensor_metadata(item))
                else:
                    metas.append(item)
            out[node.name] = tuple(metas)
    return out


def _inductor_backend_collecting_fx_meta(captured: dict[str, object]):
    def backend(gm, example_inputs):
        captured.update(_collect_fx_tensor_meta(gm))
        return compile_fx(gm, example_inputs)

    return backend


@requires_npu
class TestTeFakeTensorCacheOnOffCrosscheck(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        FakeTensorMode.cache_clear()

    def test_cache_on_off_crosscheck(self):
        def fn(x):
            return (x * 2).sin() + 1

        x = torch.randn(4, 8, device="npu")

        def run(*, cache_enabled: bool, crosscheck: bool) -> dict[str, object]:
            torch._dynamo.reset()
            FakeTensorMode.cache_clear()
            captured: dict[str, object] = {}
            compiled = torch.compile(
                fn,
                backend=_inductor_backend_collecting_fx_meta(captured),
                fullgraph=True,
            )
            with torch._dynamo.config.patch(
                fake_tensor_cache_enabled=cache_enabled,
                fake_tensor_cache_crosscheck_enabled=crosscheck,
            ):
                compiled(x)
            return captured

        meta_off = run(cache_enabled=False, crosscheck=False)
        meta_on = run(cache_enabled=True, crosscheck=True)
        self.assertTrue(meta_off, "expected FX tensor metadata with cache off")
        self.assertEqual(meta_off, meta_on)


if __name__ == "__main__":
    run_tests()
