# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.sym_node.is_contiguous 接口功能正确性
API 名称：torch.fx.experimental.sym_node.is_contiguous
API 签名：is_contiguous(sizes, strides) -> SymInt | SymFloat | SymBool | int | float | bool

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的零尺寸或 rank 不匹配；专用布局不构造无意义的零 rank |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：整数与符号尺寸、步长列表及显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：零尺寸、零 rank 与符号尺寸 |
| 正常传参场景 | 已覆盖：连续、转置与单位维度步长 |
| 异常传参场景 | 不适用：有效尺寸/步长组合以布尔或整数结果表达 |
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

from torch.fx.experimental.sym_node import is_contiguous
from torch.fx.experimental.symbolic_shapes import ShapeEnv


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_is_contiguous_contiguous_transpose_and_size_one(self):
        self.assertTrue(bool(is_contiguous([2, 3], [3, 1])))
        self.assertFalse(bool(is_contiguous([2, 3], [1, 2])))
        self.assertTrue(bool(is_contiguous([2, 1, 3], [3, 99, 1])))

    def test_is_contiguous_empty_and_symbolic_sizes(self):
        self.assertTrue(bool(is_contiguous([0, 3], [99, 4])))
        self.assertTrue(bool(is_contiguous([], [])))
        env = ShapeEnv()
        self.addCleanup(env.cleanup)
        symbolic = env.create_unbacked_symint()
        before = list(env.guards)
        self.assertTrue(bool(is_contiguous([symbolic, 3], [3, 1])))
        self.assertEqual(env.guards, before)


if __name__ == "__main__":
    run_tests()
