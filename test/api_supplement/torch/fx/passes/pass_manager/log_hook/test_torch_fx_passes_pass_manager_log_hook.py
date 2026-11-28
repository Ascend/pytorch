# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.pass_manager.log_hook 接口功能正确性
API 名称：torch.fx.passes.pass_manager.log_hook
API 签名：log_hook(fn, level=logging.INFO) -> collections.abc.Callable[~_P, ~_R]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空支持链、初始状态或无迭代边界 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 FX 节点或带可观测状态的 Python pass 回调 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：异常回调无结果日志 |
| 正常传参场景 | 已覆盖：结果保留、默认 INFO 与显式 DEBUG 日志 |
| 异常传参场景 | 已覆盖：回调 ValueError 传播且日志捕获上下文恢复 logger 状态 |
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

import logging
from torch.fx.passes import pass_manager
from torch.fx.passes.pass_manager import log_hook


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_log_hook_output_and_explicit_log_level(self):
        def combine(left, *, right=2):
            return left + right

        with self.assertLogs(pass_manager.logger, level=logging.DEBUG) as captured:
            output = log_hook(combine, level=logging.DEBUG)(3, right=4)
        self.assertEqual(output, 7)
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].levelno, logging.DEBUG)
        self.assertIn("7", captured.records[0].getMessage())

    def test_log_hook_default_level_and_callback_exception(self):
        with self.assertLogs(pass_manager.logger, level=logging.INFO) as captured:
            self.assertEqual(log_hook(lambda: "result")(), "result")
        self.assertEqual(captured.records[0].levelno, logging.INFO)

        def fail():
            raise ValueError("logging pass failed")

        with self.assertNoLogs(pass_manager.logger, level=logging.INFO):
            with self.assertRaisesRegex(ValueError, "logging pass failed"):
                log_hook(fail)()


if __name__ == "__main__":
    run_tests()
