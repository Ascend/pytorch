# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.has_symbolic_sizes_strides 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.has_symbolic_sizes_strides
API 签名：has_symbolic_sizes_strides(elem) -> bool

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空状态、约束或记录 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 ShapeEnv、SymPy 表达式及对应默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：具体元数据、标量、零维长度与静态 FakeTensor |
| 正常传参场景 | 已覆盖：真实 FakeTensor 符号 shape 标记 |
| 异常传参场景 | 未覆盖：无稳定异常契约 |
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

from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv, has_symbolic_sizes_strides


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_has_symbolic_sizes_strides_concrete_and_scalar_metadata_are_not_symbolic(self):
        for shape in ((2, 3), (), (0, 4)):
            with self.subTest(shape=shape):
                self.assertFalse(has_symbolic_sizes_strides(torch.empty(shape, device="meta")))

    def test_has_symbolic_sizes_strides_real_fake_tensor_symbolic_dimensions_are_detected(self):
        env = ShapeEnv()
        self.addCleanup(env.cleanup)
        mode = FakeTensorMode(shape_env=env)
        fake = mode.from_tensor(torch.empty(2, 3), static_shapes=False)
        self.assertTrue(any(isinstance(size, torch.SymInt) for size in fake.shape))
        self.assertTrue(has_symbolic_sizes_strides(fake))
        static = mode.from_tensor(torch.empty(4, 5), static_shapes=True)
        self.assertFalse(has_symbolic_sizes_strides(static))


if __name__ == "__main__":
    run_tests()
