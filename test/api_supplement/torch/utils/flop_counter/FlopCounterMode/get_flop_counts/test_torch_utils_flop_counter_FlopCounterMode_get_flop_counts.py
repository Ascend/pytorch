# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.flop_counter.FlopCounterMode.get_flop_counts 接口功能正确性
API 名称：torch.utils.flop_counter.FlopCounterMode.get_flop_counts
API 签名：FlopCounterMode.get_flop_counts(self)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 空上下文返回空字典与含算子上下文非空统计   | 已覆盖（empty_context、mm_counts_cpu_and_device）      |
| 枚举选项         | 接口枚举类参数覆盖                         | 未覆盖（接口无枚举类参数）                             |
| 参数类型         | 不同入参类型覆盖                           | 未覆盖（无参实例方法，仅接收 self）                    |
| 传参与不传参     | 可选默认参数省略与显式传入                 | 未覆盖（无可选参数，不接受额外入参）                   |
| 等价类/边界值    | 零算子边界与返回值深拷贝隔离内部计数       | 已覆盖（empty_context、returns_copy_of_internal_counts） |
| 正常传参场景     | CPU/目标设备 mm 与模块级 Linear 统计       | 已覆盖（mm_counts_cpu_and_device、module_level_tracking） |
| 异常传参场景     | 传入多余位置参数触发 TypeError             | 已覆盖（unexpected_arg_raises）                        |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（无参统计查询方法，算子已在两侧分别验证）       |

未覆盖项及原因：
- 枚举选项：接口无枚举类参数。
- 参数类型：get_flop_counts 为无参实例方法，不接收除 self 外的入参。
- 传参与不传参：无可选默认参数。
- 混合设备类型：查询方法不接收 Tensor 输入；算子统计已在 CPU 与目标设备分别验证。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

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
class TestFlopCounterModeGetFlopCounts(TestCase):
    """Test cases for torch.utils.flop_counter.FlopCounterMode.get_flop_counts."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_mm_counts_cpu_and_device(self):
        """Records exact torch.mm FLOP counts under Global on CPU and target device."""
        m, k, n = 4, 6, 5
        expected = 2 * m * k * n
        for dev in (torch.device("cpu"), self.device):
            a = torch.randn(m, k, device=dev)
            b = torch.randn(k, n, device=dev)
            with FlopCounterMode(display=False) as fc:
                _ = torch.mm(a, b)
            counts = fc.get_flop_counts()
            self.assertIsInstance(counts, dict)
            self.assertIn("Global", counts)
            self.assertEqual(counts["Global"][torch.ops.aten.mm], expected)

    def test_module_level_tracking(self):
        """Tracks FLOP counts per nn.Module entry in addition to Global."""
        batch, in_features, out_features = 3, 8, 5
        expected = 2 * batch * in_features * out_features
        linear = nn.Linear(in_features, out_features).to(self.device)
        x = torch.randn(batch, in_features, device=self.device)
        with FlopCounterMode(display=False) as fc:
            _ = linear(x)
        counts = fc.get_flop_counts()
        self.assertIn("Global", counts)
        self.assertEqual(sum(counts["Global"].values()), expected)
        module_keys = [k for k in counts if k != "Global"]
        self.assertGreaterEqual(len(module_keys), 1)
        for mod_key in module_keys:
            self.assertEqual(sum(counts[mod_key].values()), expected)

    def test_empty_context(self):
        """Returns an empty dict when no ops are executed inside the context."""
        with FlopCounterMode(display=False) as fc:
            pass
        counts = fc.get_flop_counts()
        self.assertEqual(counts, {})

    def test_returns_copy_of_internal_counts(self):
        """Mutating the returned dict does not affect subsequent get_flop_counts calls."""
        a = torch.randn(2, 3, device=self.device)
        b = torch.randn(3, 4, device=self.device)
        with FlopCounterMode(display=False) as fc:
            _ = torch.mm(a, b)
        counts = fc.get_flop_counts()
        counts["Global"][torch.ops.aten.mm] = -1
        counts["Injected"] = {}
        fresh = fc.get_flop_counts()
        self.assertNotIn("Injected", fresh)
        self.assertEqual(fresh["Global"][torch.ops.aten.mm], 2 * 2 * 3 * 4)

    def test_unexpected_arg_raises(self):
        """Passing an unexpected argument to get_flop_counts raises TypeError."""
        fc = FlopCounterMode(display=False)
        with self.assertRaises(TypeError):
            fc.get_flop_counts("unexpected")


if __name__ == "__main__":
    run_tests()
