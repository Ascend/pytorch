# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.compiler.save_cache_artifacts 接口功能正确性
API 名称：torch.compiler.save_cache_artifacts
API 签名：save_cache_artifacts() -> tuple[bytes, torch.compiler._cache.CacheInfo] | None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：无产物或空字节 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：默认调用与 bytes 产物 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：两组输入、空/无效产物及全局 manager 恢复 |
| 正常传参场景 | 已覆盖：真实 Inductor 编译缓存产物序列化/热加载 |
| 异常传参场景 | 未覆盖：没有稳定的异常契约 |
| 混合设备类型 | 未覆盖：单加速器运行 |

未覆盖项及原因：
- 图执行方式：直接调用 save/load；产物来自实际 Inductor/fullgraph 编译。
- eager 对照：真实加速器、两组不同输入，与同设备 eager 对照。
- autograd：未覆盖 backward 缓存。
- RNG：确定输入。
- 参数类型：未覆盖跨进程、跨硬件和跨版本缓存迁移。
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

import torch._dynamo
from torch.compiler._cache import CacheArtifactManager


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _compile_and_serialize(self):
        self.assertIsNotNone(self.device)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

        def function(value):
            return value.cos() + value * value + 3.125

        compiled = torch.compile(function, backend="inductor", fullgraph=True, dynamic=False)
        value = torch.arange(6, dtype=torch.float32, device=self.device).reshape(2, 3) / 7
        for current in (value, value + 0.5):
            actual = compiled(current)
            torch.testing.assert_close(actual.cpu(), function(current).cpu())
            self.assertEqual(actual.shape, current.shape)
            self.assertEqual(actual.dtype, current.dtype)
            self.assertEqual(actual.device.type, self.device_name)
        saved = torch.compiler.save_cache_artifacts()
        self.assertIsNotNone(saved)
        payload, info = saved
        self.assertIsInstance(payload, bytes)
        self.assertGreater(len(payload), 0)
        self.assertTrue(info.inductor_artifacts)
        return function, value, payload, info

    def test_save_cache_artifacts_serialized_real_inductor_artifacts_and_repeated_save(self):
        original = CacheArtifactManager._cache_info
        with CacheArtifactManager.with_fresh_cache():
            with torch._dynamo.config.patch(caching_precompile=False):
                _, _, _, info = self._compile_and_serialize()
                again = torch.compiler.save_cache_artifacts()
                self.assertIsNotNone(again)
                self.assertEqual(again[1].artifacts, info.artifacts)
                self.assertGreater(len(again[0]), 0)
                self.assertFalse(CacheArtifactManager.need_serialize())
        self.assertIs(CacheArtifactManager._cache_info, original)

    def test_save_cache_artifacts_no_artifacts_returns_none_and_exception_restores_manager(self):
        original = CacheArtifactManager._cache_info
        with self.assertRaisesRegex(RuntimeError, "controlled"):
            with CacheArtifactManager.with_fresh_cache():
                with torch._dynamo.config.patch(caching_precompile=False):
                    self.assertIsNone(torch.compiler.save_cache_artifacts())
                    raise RuntimeError("controlled")
        self.assertIs(CacheArtifactManager._cache_info, original)


if __name__ == "__main__":
    run_tests()
