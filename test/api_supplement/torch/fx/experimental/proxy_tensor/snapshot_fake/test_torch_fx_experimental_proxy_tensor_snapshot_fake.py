# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.proxy_tensor.snapshot_fake 接口功能正确性
API 名称：torch.fx.experimental.proxy_tensor.snapshot_fake
API 签名：snapshot_fake(val, include_real=False) -> Tensor | None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 适用的无 mode、空状态或空参数 |
| 枚举选项 | include_real=False/True |
| 参数类型 | FakeTensor、非 fake meta Tensor 和携带 real_tensor 的 FakeTensor |
| 传参与不传参 | include_real 省略及显式两值 |
| 等价类/边界值 | 零尺寸、detach、独立元数据及 real_tensor 保留/移除 |
| 正常传参场景 | 真实 FakeTensor 快照、detach 及独立 metadata |
| 异常传参场景 | 未覆盖：无稳定异常契约 |
| 混合设备类型 | 不适用：本文件为设备无关的图结构、类型或辅助状态语义 |

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
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_snapshot_fake_fake_snapshot_detaches_gradient_and_keeps_independent_metadata(self):
        mode = FakeTensorMode()
        with mode:
            value = torch.empty_strided((2, 3), (1, 2), requires_grad=True)
            snapshot = pt.snapshot_fake(value)
            self.assertIsNot(snapshot, value)
            self.assertTrue(is_fake(snapshot))
            self.assertFalse(snapshot.requires_grad)
            self.assertTrue(value.requires_grad)
            self.assertEqual(snapshot.shape, (2, 3))
            self.assertEqual(snapshot.stride(), (1, 2))
            with torch.no_grad():
                value.transpose_(0, 1)
            self.assertEqual(snapshot.shape, (2, 3))
            self.assertEqual(value.shape, (3, 2))

    def test_snapshot_fake_non_fake_metadata_tensor_is_detached_without_shape_change(self):
        value = torch.empty((0, 4), device="meta", requires_grad=True)
        result = pt.snapshot_fake(value, include_real=True)
        self.assertIsNot(result, value)
        self.assertEqual(result.shape, value.shape)
        self.assertEqual(result.dtype, value.dtype)
        self.assertFalse(result.requires_grad)


    def test_parameter_include_real_false_and_true_on_real_backed_fake_tensor(self):
        from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
        from torch._functorch import config

        real = torch.ones(2, 3, device=self.device, requires_grad=True)
        with config.patch(fake_tensor_propagate_real_tensors=True):
            mode = FakeTensorMode()
            fake = mode.from_tensor(real)
            self.assertIsNotNone(fake.real_tensor)
            without_real = pt.snapshot_fake(fake, include_real=False)
            with_real = pt.snapshot_fake(fake, include_real=True)
            self.assertTrue(is_fake(with_real))
            self.assertIsNot(with_real, fake)
            self.assertFalse(with_real.requires_grad)
            self.assertEqual(with_real.shape, (2, 3))
            self.assertIsNone(without_real.real_tensor)
            self.assertIsNotNone(with_real.real_tensor)
            torch.testing.assert_close(with_real.real_tensor.cpu(), real.detach().cpu())


if __name__ == "__main__":
    run_tests()
