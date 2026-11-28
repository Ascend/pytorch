# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.compiler.precompile.MakeFxTracer 接口功能正确性
API 名称：torch.compiler.precompile.MakeFxTracer
API 签名：MakeFxTracer(decompositions=None)

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的未调用/未激活捕获状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 Capture/MakeFxTracer、Path 产物、Inductor 后端和设备输入 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：默认表及显式 decomposition 的实际执行 |
| 正常传参场景 | 已覆盖：实际 MakeFxTracer 配置、不可变字段和完整捕获 |
| 异常传参场景 | 已覆盖：冻结配置赋值异常 |
| 混合设备类型 | 未覆盖：同设备输入，未覆盖跨设备产物迁移 |

未覆盖项及原因：
- 图执行方式：直接调用 precompile 入口；产物 BACKEND 核验为 Inductor，不在外层再套 torch.compile。
- eager 对照：同设备、同输入；加载产物后对照两组不同输入。
- autograd：未覆盖预编译 backward 产物。
- RNG：确定输入，临时产物和 runnable 在用例结束清理。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
DEVICE_NAME = None
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch_npu":
            raise
    else:
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    if DEVICE_NAME is None and torch.cuda.is_available():
        DEVICE_NAME = "cuda"

if DEVICE_NAME == "npu":
    from torch_npu.testing.testcase import TestCase, run_tests
else:
    from unittest import TestCase

    def run_tests():
        unittest.main()

import ast
from pathlib import Path
import tempfile
import torch._dynamo
from torch.compiler import precompile as pc

import dataclasses


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    @staticmethod
    def _function(value):
        return value.sin() + value * value

    def _paths(self):
        directory = tempfile.TemporaryDirectory(prefix="graph-ut-precompile-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        return root / "artifact.py", root / "artifact.cache"

    def _value(self):
        self.assertIsNotNone(self.device)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        return torch.arange(6, dtype=torch.float32, device=self.device).reshape(2, 3) / 5

    def _capture_pair(self, tracer=None):
        artifact, cache = self._paths()
        value = self._value()
        kwargs = {"tracer": tracer} if tracer is not None else {}
        with pc.capture(self._function, artifact_path=artifact, cache_path=cache, backend="inductor", **kwargs) as cap:
            result = cap(value)
            torch.testing.assert_close(result.cpu(), self._function(value).cpu())
        self.assertTrue(artifact.is_file())
        self.assertTrue(cache.is_file())
        assignments = [node for node in ast.parse(artifact.read_text()).body if isinstance(node, ast.Assign)]
        backend_values = [ast.literal_eval(node.value) for node in assignments
                          if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                          and node.targets[0].id == "BACKEND"]
        self.assertEqual(backend_values, ["inductor"])
        runnable = pc.load(artifact, cache)
        self.addCleanup(runnable.unload)
        return cap, runnable, value, artifact, cache

    def _assert_loaded(self, runnable, value):
        for current in (value, value + 0.4):
            actual, expected = runnable(current), self._function(current)
            torch.testing.assert_close(actual.cpu(), expected.cpu())
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.device.type, self.device_name)

    def test_makefxtracer_default_configuration_is_frozen_and_usable_for_real_capture(self):
        tracer = pc.MakeFxTracer()
        self.assertIsNone(tracer.decompositions)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            tracer.decompositions = {}
        _, runnable, value, _, _ = self._capture_pair(tracer)
        self._assert_loaded(runnable, value)

    def test_makefxtracer_explicit_decomposition_is_applied_during_capture(self):
        observed = []

        def negate(value):
            observed.append("decomposed")
            return value * -1

        def function(value):
            return torch.ops.aten.neg.default(value) + 1

        tracer = pc.MakeFxTracer(decompositions={torch.ops.aten.neg.default: negate})
        artifact, cache = self._paths()
        value = self._value()
        with pc.capture(function, artifact_path=artifact, cache_path=cache, tracer=tracer, backend="inductor") as cap:
            result = cap(value)
            torch.testing.assert_close(result.cpu(), (-value + 1).cpu())
        self.assertTrue(observed)
        with pc.load(artifact, cache) as runnable:
            for current in (value, value + 0.4):
                torch.testing.assert_close(runnable(current).cpu(), (-current + 1).cpu())


if __name__ == "__main__":
    run_tests()
