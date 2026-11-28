# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.compiler.load_cache_artifacts 接口功能正确性
API 名称：torch.compiler.load_cache_artifacts
API 签名：load_cache_artifacts(serialized_artifacts) -> torch.compiler._cache.CacheInfo | None

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

    def test_load_cache_artifacts_hot_load_reports_real_cache_keys_and_recompiled_graph_matches_eager(self):
        with CacheArtifactManager.with_fresh_cache():
            with torch._dynamo.config.patch(caching_precompile=False):
                function, value, payload, saved_info = self._compile_and_serialize()
                loaded_info = torch.compiler.load_cache_artifacts(payload)
                self.assertIsNotNone(loaded_info)
                self.assertEqual(loaded_info.artifacts, saved_info.artifacts)
                torch._dynamo.reset()
                compiled = torch.compile(function, backend="inductor", fullgraph=True, dynamic=False)
                for current in (value, value + 0.7):
                    actual = compiled(current)
                    torch.testing.assert_close(actual.cpu(), function(current).cpu())
                    self.assertEqual(actual.device.type, self.device_name)

    def test_load_cache_artifacts_load_populates_an_empty_fx_graph_cache_directory(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        from torch._inductor.codecache import FxGraphCache

        with CacheArtifactManager.with_fresh_cache():
            with torch._dynamo.config.patch(caching_precompile=False):
                _, _, payload, saved_info = self._compile_and_serialize()
                with tempfile.TemporaryDirectory(prefix="graph-ut-cold-load-") as directory:
                    destination = Path(directory).resolve()
                    self.assertEqual(list(destination.iterdir()), [])
                    with mock.patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": directory}):
                        loaded_info = torch.compiler.load_cache_artifacts(payload)
                        self.assertIsNotNone(loaded_info)
                        self.assertEqual(loaded_info.artifacts, saved_info.artifacts)
                        self.assertTrue(loaded_info.inductor_artifacts)
                        for key in loaded_info.inductor_artifacts:
                            cache = Path(FxGraphCache._get_tmp_dir_for_key(key)).resolve()
                            self.assertTrue(cache.is_relative_to(destination))
                            self.assertTrue(cache.is_dir(), "load did not populate the serialized graph key")
                            entries = [entry for entry in cache.iterdir() if entry.is_file()]
                            self.assertTrue(entries)
                            self.assertTrue(all(entry.stat().st_size > 0 for entry in entries))

    def test_load_cache_artifacts_invalid_serialized_bytes_return_none_without_recording_artifacts(self):
        with CacheArtifactManager.with_fresh_cache():
            for payload in (b"", b"invalid-cache-format"):
                with self.subTest(payload=payload):
                    self.assertIsNone(torch.compiler.load_cache_artifacts(payload))
            self.assertFalse(CacheArtifactManager.need_serialize())
            self.assertTrue(CacheArtifactManager._cache_info.empty())


if __name__ == "__main__":
    run_tests()
