# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.proxy_tensor.disable_autocast_cache 接口功能正确性
API 名称：torch.fx.experimental.proxy_tensor.disable_autocast_cache
API 签名：disable_autocast_cache() -> Generator[None, None, None]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的无 mode、空状态或空参数 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 tracer/mode、Tensor 或符号及默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：正常/异常退出恢复 |
| 正常传参场景 | 已覆盖：真实 autocast 缓存状态与嵌套作用域 |
| 异常传参场景 | 已覆盖：受控异常后缓存恢复 |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
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

from torch.fx.experimental import proxy_tensor as pt


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_disable_autocast_cache_nested_context_disables_cache_and_restores_original(self):
        original = torch.is_autocast_cache_enabled()
        self.addCleanup(torch.set_autocast_cache_enabled, original)
        torch.set_autocast_cache_enabled(True)
        with pt.disable_autocast_cache():
            self.assertFalse(torch.is_autocast_cache_enabled())
            with pt.disable_autocast_cache():
                self.assertFalse(torch.is_autocast_cache_enabled())
            self.assertFalse(torch.is_autocast_cache_enabled())
        self.assertTrue(torch.is_autocast_cache_enabled())
        torch.set_autocast_cache_enabled(original)
        self.assertEqual(torch.is_autocast_cache_enabled(), original)

    def test_disable_autocast_cache_exception_exit_restores_cache(self):
        original = torch.is_autocast_cache_enabled()
        with self.assertRaisesRegex(RuntimeError, "controlled"):
            with pt.disable_autocast_cache():
                self.assertFalse(torch.is_autocast_cache_enabled())
                raise RuntimeError("controlled")
        self.assertEqual(torch.is_autocast_cache_enabled(), original)


if __name__ == "__main__":
    run_tests()
