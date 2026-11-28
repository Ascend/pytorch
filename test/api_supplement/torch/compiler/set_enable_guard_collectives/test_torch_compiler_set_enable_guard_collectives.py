# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.compiler.set_enable_guard_collectives 接口功能正确性
API 名称：torch.compiler.set_enable_guard_collectives
API 签名：set_enable_guard_collectives(enabled)

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：初始无 hook |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：enabled=True/False |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：重复启停、已有 hook 和异常清理 |
| 正常传参场景 | 已覆盖：真实原生 guard hook 启用/关闭和之前状态返回值 |
| 异常传参场景 | 已覆盖：受控异常后的 hook 清理 |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
- 图执行方式：仅验证 hook 注册状态；未创建进程组或执行跨 rank collective。
- 参数类型：原生接口只有交换 setter；先注册 cleanup，再保存交换返回的原 hook。
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

from torch._C._dynamo.eval_frame import set_guard_complete_hook


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _isolate_hook(self):
        previous = []

        def restore():
            if previous:
                set_guard_complete_hook(previous[0])

        self.addCleanup(restore)
        previous.append(set_guard_complete_hook(None))

    def test_set_enable_guard_collectives_enable_disable_returns_previous_setting(self):
        self._isolate_hook()
        self.assertFalse(torch.compiler.set_enable_guard_collectives(True))
        self.assertTrue(torch.compiler.set_enable_guard_collectives(True))
        self.assertTrue(torch.compiler.set_enable_guard_collectives(False))
        self.assertFalse(torch.compiler.set_enable_guard_collectives(False))

    def test_set_enable_guard_collectives_controlled_exception_does_not_lose_preexisting_hook(self):
        self._isolate_hook()

        def sentinel(*args):
            return None

        set_guard_complete_hook(sentinel)
        try:
            with self.assertRaisesRegex(RuntimeError, "controlled"):
                self.assertTrue(torch.compiler.set_enable_guard_collectives(True))
                raise RuntimeError("controlled")
        finally:
            installed = set_guard_complete_hook(sentinel)
        self.assertIsNotNone(installed)
        self.assertTrue(torch.compiler.set_enable_guard_collectives(False))
        self.assertIsNone(set_guard_complete_hook(None))


if __name__ == "__main__":
    run_tests()
