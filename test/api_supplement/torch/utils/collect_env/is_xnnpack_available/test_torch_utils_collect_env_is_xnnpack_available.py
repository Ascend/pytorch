# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.collect_env.is_xnnpack_available 接口功能正确性
API 名称：torch.utils.collect_env.is_xnnpack_available
API 签名：torch.utils.collect_env.is_xnnpack_available()

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空与非空输入                         | 未覆盖（见下）                                     |
| 枚举选项         | 返回值 'True' / 'False' / 'N/A' 状态 | 已覆盖（tracks_mocked_enabled_flag、torch_unavailable_returns_na） |
| 参数类型         | 输入参数类型                         | 未覆盖（见下）                                     |
| 传参与不传参     | 无参调用与多余位置/关键字参数        | 已覆盖（matches_xnnpack_enabled、unexpected_arg_raises） |
| 等价类/边界值    | 多次调用一致性与 TORCH_AVAILABLE=False | 已覆盖（deterministic_across_calls、torch_unavailable_returns_na） |
| 正常传参场景     | 查询 XNNPACK 可用性及 mock 状态切换  | 已覆盖（matches_xnnpack_enabled、tracks_mocked_enabled_flag 等） |
| 异常传参场景     | 传入多余位置或关键字参数报 TypeError | 已覆盖（unexpected_arg_raises）                    |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 空/非空：无参查询接口，不涉及空/非空输入。
- 参数类型：接口不接收外部参数，多余参数已在异常传参场景验证。
- 混合设备类型：环境检测辅助函数，不接收 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest
from unittest.mock import patch

import torch
import torch.backends.xnnpack
import torch.utils.collect_env as collect_env
from torch.utils.collect_env import is_xnnpack_available

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe device only when RUN_FLAG is enabled.
DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestIsXnnpackAvailable(TestCase):
    """Test cases for torch.utils.collect_env.is_xnnpack_available."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_matches_xnnpack_enabled(self):
        """Returns string matching str(torch.backends.xnnpack.enabled) when torch is available."""
        result = is_xnnpack_available()
        self.assertIsInstance(result, str)
        self.assertIn(result, ("True", "False"))
        self.assertEqual(result, str(torch.backends.xnnpack.enabled))

    def test_deterministic_across_calls(self):
        """Repeated calls return identical status strings."""
        first = is_xnnpack_available()
        second = is_xnnpack_available()
        self.assertEqual(first, second)

    def test_torch_unavailable_returns_na(self):
        """Returns 'N/A' when collect_env.TORCH_AVAILABLE is False."""
        with patch.object(collect_env, "TORCH_AVAILABLE", False):
            self.assertEqual(is_xnnpack_available(), "N/A")

    def test_tracks_mocked_enabled_flag(self):
        """Reflects toggled torch.backends.xnnpack.enabled values accurately."""
        from types import SimpleNamespace

        with patch.object(collect_env.torch.backends, "xnnpack", SimpleNamespace(enabled=True)):
            self.assertEqual(is_xnnpack_available(), "True")
        with patch.object(collect_env.torch.backends, "xnnpack", SimpleNamespace(enabled=False)):
            self.assertEqual(is_xnnpack_available(), "False")

    def test_unexpected_arg_raises(self):
        """Passing unexpected positional or keyword args raises TypeError."""
        with self.assertRaises(TypeError):
            is_xnnpack_available(True)
        with self.assertRaises(TypeError):
            is_xnnpack_available(flag=True)


if __name__ == "__main__":
    run_tests()
