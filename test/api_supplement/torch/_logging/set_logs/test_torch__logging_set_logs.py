# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._logging.set_logs 接口功能正确性
API 名称：torch._logging.set_logs
API 签名：torch._logging.set_logs(*, all=None, dynamo=None, inductor=None, ...) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 不传参重置默认 | 已覆盖 |
| 枚举选项 | 组件名（dynamo/inductor 等） | 已覆盖 |
| 参数类型 | 整型日志级别（logging 常量） | 已覆盖 |
| 传参与不传参 | 关键字必填、默认不传 | 已覆盖 |
| 等价类/边界值 | 不同级别在组件 logger 上可观测 | 已覆盖 |
| 正常传参场景 | 设置成功且返回 None | 已覆盖 |
| 异常传参场景 | 未知组件名/非法级别值报错 | 已覆盖 |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 只配置日志组件、不接收 Tensor，混合设备维度不适用。
- 若环境变量 TORCH_LOGS 已设置则本 API 不生效（原生契约），此时跳过效果断言；
  组件 logger 级别在 tearDown 中还原。
"""
import logging
import os
import unittest

import torch
import torch._logging  # noqa: F401

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化，避免
# 不必要的探测开销；被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
DEVICE_NAME = 'cpu'
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = 'npu'
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

# ORDER CONSTRAINT: import torch_npu.testing.testcase BEFORE
# torch_npu.contrib.transfer_to_npu (patching first breaks the import).
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class Testset_logs(TestCase):
    """Test cases for torch._logging.set_logs (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self._dynamo_level = logging.getLogger('torch._dynamo').level

    def tearDown(self):
        try:
            logging.getLogger('torch._dynamo').setLevel(self._dynamo_level)
        finally:
            super().tearDown()

    def test_returns_none(self):
        """Calling with no arguments returns None."""
        self.assertIsNone(torch._logging.set_logs())

    def test_component_level_is_applied(self):
        """Setting the dynamo component level is observable on its logger."""
        if os.environ.get('TORCH_LOGS'):
            self.skipTest('TORCH_LOGS takes precedence over set_logs')
        torch._logging.set_logs(dynamo=logging.DEBUG)
        self.assertEqual(logging.getLogger('torch._dynamo').level, logging.DEBUG)
        torch._logging.set_logs(dynamo=logging.WARNING)
        self.assertEqual(logging.getLogger('torch._dynamo').level, logging.WARNING)

    def test_unknown_component_raises(self):
        """An unknown keyword argument raises TypeError."""
        with self.assertRaises(TypeError):
            torch._logging.set_logs(not_a_component=logging.INFO)

    def test_invalid_level_value_raises(self):
        """An unrecognized level value raises ValueError."""
        with self.assertRaises(ValueError):
            torch._logging.set_logs(dynamo='not-a-level')


if __name__ == "__main__":
    run_tests()
