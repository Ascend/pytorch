# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.Kernel.index 接口功能正确性
API 名称：torch.autograd.profiler_util.Kernel.index
API 签名：Kernel.index(self, value, start=0, stop=sys.maxsize)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 容器空与非空状态 | 未覆盖，Kernel 为固定 3 字段 namedtuple，无空实例 |
| 枚举选项 | 枚举参数各合法取值 | 未覆盖，该接口无枚举参数 |
| 参数类型 | 不同类型字段值查找 | 已覆盖（find_all_fields 覆盖 str/int/float） |
| 传参与不传参 | start/stop 可选参数省略与传入 | 已覆盖（duplicate_with_bounds） |
| 等价类/边界值 | 重复元素、负索引切片与切片外边界 | 已覆盖（duplicate_with_bounds、negative_start_stop、out_of_slice_raises） |
| 正常传参场景 | 标准字段及范围查找 | 已覆盖（find_all_fields、duplicate_with_bounds、negative_start_stop） |
| 异常传参场景 | 缺失值、越界切片及缺参异常 | 已覆盖（missing_value_raises、out_of_slice_raises、missing_arg_raises） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该接口为纯 Python namedtuple 方法，不涉及设备迁移 |

未覆盖项及原因：
- 空/非空：Kernel 为固定三字段 namedtuple，不支持构造空容器。
- 枚举选项：该接口不包含枚举类型参数。
- 混合设备类型：该接口为纯 Python 元组查找方法，不接受 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.autograd.profiler_util import Kernel

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
class TestKernelIndex(TestCase):
    """Test cases for torch.autograd.profiler_util.Kernel.index."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_find_all_fields(self):
        """index locates name at 0, device at 1, and duration at 2."""
        k = Kernel(name="matmul_kernel", device=0, duration=42.5)
        self.assertEqual(k.index("matmul_kernel"), 0)
        self.assertEqual(k.index(0), 1)
        self.assertEqual(k.index(42.5), 2)

    def test_duplicate_with_bounds(self):
        """With duplicate field values, start/stop bounds select the matching occurrence."""
        k = Kernel(name="k", device=1, duration=1)
        self.assertEqual(k.index(1), 1)
        self.assertEqual(k.index(1, 2), 2)
        self.assertEqual(k.index(1, 1, 2), 1)
        self.assertEqual(k.index(1, 2, 3), 2)

    def test_negative_start_stop(self):
        """Negative start and stop indices resolve relative to tuple length."""
        k = Kernel(name="k", device=5, duration=5)
        self.assertEqual(k.index(5, -2), 1)
        self.assertEqual(k.index(5, -1), 2)
        self.assertEqual(k.index(5, -2, -1), 1)

    def test_missing_value_raises(self):
        """Searching for a value absent from Kernel raises ValueError."""
        k = Kernel(name="conv_kernel", device=0, duration=12.0)
        with self.assertRaises(ValueError):
            k.index("nonexistent")
        with self.assertRaises(ValueError):
            k.index(999)

    def test_out_of_slice_raises(self):
        """Value present in Kernel but outside [start, stop) raises ValueError."""
        k = Kernel(name="conv_kernel", device=0, duration=12.0)
        with self.assertRaises(ValueError):
            k.index("conv_kernel", 1)
        with self.assertRaises(ValueError):
            k.index(12.0, 0, 2)
        with self.assertRaises(ValueError):
            k.index(0, 2, 3)

    def test_missing_arg_raises(self):
        """Calling index without value raises TypeError."""
        k = Kernel(name="conv_kernel", device=0, duration=12.0)
        with self.assertRaises(TypeError):
            k.index()


if __name__ == "__main__":
    run_tests()
