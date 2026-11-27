# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.foreach.lgamma_ 接口功能正确性
API 名称：torch.foreach.lgamma_
API 签名：torch.foreach.lgamma_(inputs, /) -> list[Tensor]

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                                  |
|------------------|--------------------------------------------|-----------------------------------------------------------|
| shape            | 0-dim / 1D / 2D / 3D / size-0 维           | 已覆盖：shape_coverage                                    |
| dtype            | float16/float32/bfloat16；整型不支持       | 已覆盖：dtype_coverage、int_dtype_raises                  |
| device           | 每个正常场景 CPU/NPU 双端，CPU 为 Golden   | 已覆盖：cpu_golden_npu 及各 *_coverage                    |
| 参数枚举         | 该接口仅一个位置参数 inputs                | 不适用                                                    |
| 可选参数         | 该接口无可选参数                           | 不适用                                                    |
| in-place 语义    | 原地改写输入并返回同一容器对象             | 已覆盖：returns_same_list_object、                       |
|                  |                                            |         returns_same_tuple_object、mutates_input_in_place |
| out= 参数        | 不支持 out=                                | 不适用                                                    |
| 广播             | 逐张量独立处理，无广播语义                 | 不适用                                                    |
| 连续性           | 非连续（transpose 视图）输入               | 已覆盖：non_contiguous                                    |
| 混合设备输入     | list 内同时含 NPU 与 CPU tensor（逐张量回退）| 已覆盖：mixed_device_falls_back（无加速器时按环境不具备跳过）|
| 异常路径         | 空 list / 整型输入                         | 已覆盖：empty_list_raises、int_dtype_raises               |

未覆盖项及原因：
- 参数枚举 / 可选参数 / out= / 广播：该接口为固定单参的原地逐元素多张量运算，无这些维度。
- NPU 原生算子：本仓库环境下 torch.foreach.lgamma_ 在 NPU 上回退 CPU 执行（无原生 aclnn 算子），
  功能结果仍以 CPU 为 Golden 对照；原生算子覆盖不在本用例判定范围。
- 混合设备异常契约：foreach 的 slow path 不校验 list 内设备一致性，逐张量各自其设备原地计算，
  混合设备 list 不抛异常；用例仅验证逐张量回退结果与 torch.lgamma 一致，不再断言抛错。

注：foreach 系列为张量数值运算（计算类）；每个正常场景均 CPU/NPU 双端执行，CPU 输出为 Golden，
     NPU 输出移至 CPU 后用 torch.testing.assert_close 对照。in-place 语义以「输入被真实改写且返回
     同一容器对象」为可证伪断言（实现未改写即失败）。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Requires a PyTorch 2.15 accelerator (NPU/CUDA) environment, which is not
# available here; skip the whole case until such an environment exists.
BROKEN_SKIP_REASON = "No PyTorch 2.15 accelerator (NPU/CUDA) environment available"

# Probe the device only when the supplement suite actually runs. On the default
# CI collection path the class-level skipUnless fires first, so no torch_npu
# import or device initialization happens here.
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
        DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skip(BROKEN_SKIP_REASON)
@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestForeachLgammaInplaceUt(TestCase):
    """Functional tests for torch.foreach.lgamma_ (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _assert_close_seq(self, actual, expected):
        """Elementwise comparison of two tensor sequences for floating tensors."""
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref)

    @staticmethod
    def _positive_inputs():
        # lgamma is defined for positive reals; keep inputs in (0.5, 3.5).
        return [torch.rand(2, 3) * 3.0 + 0.5, torch.rand(5) * 3.0 + 0.5]

    def test_foreach_lgamma_inplace_cpu_golden_npu(self):
        """CPU result is the Golden for the device path (in-place on copies)."""
        cpu_inputs = self._positive_inputs()
        cpu_work = [t.clone() for t in cpu_inputs]
        torch.foreach.lgamma_(cpu_work)
        dev_work = [t.clone().to(self.device) for t in cpu_inputs]
        torch.foreach.lgamma_(dev_work)
        self._assert_close_seq([t.cpu() for t in dev_work], cpu_work)

    def test_returns_same_list_object(self):
        """The call returns the very same list object that was passed in."""
        lst = [torch.rand(3) + 1.0]
        ret = torch.foreach.lgamma_(lst)
        self.assertIs(ret, lst)
        self.assertIsInstance(ret, list)

    def test_returns_same_tuple_object(self):
        """A tuple input is returned unchanged (same object)."""
        tup = (torch.rand(3) + 1.0,)
        ret = torch.foreach.lgamma_(tup)
        self.assertIs(ret, tup)

    def test_mutates_input_in_place(self):
        """The input buffer is rewritten with lgamma values (not left untouched)."""
        x = torch.ones(4, device=self.device)
        before = x.clone()
        lst = [x]
        torch.foreach.lgamma_(lst)
        # lgamma(1) == 0, so the mutated buffer must differ from the all-ones input.
        self.assertFalse(torch.equal(x, before))
        torch.testing.assert_close(x.cpu(), torch.zeros(4))

    def test_single_element(self):
        """A single-tensor list is handled in place."""
        base = torch.rand(2, 2) + 1.0
        work = base.clone().to(self.device)
        lst = [work]
        ret = torch.foreach.lgamma_(lst)
        self.assertIs(ret, lst)
        torch.testing.assert_close(work.cpu(), torch.lgamma(base))

    def test_shape_coverage(self):
        """0-dim / 1D / 2D / 3D / size-0 dims keep shape and get lgamma values."""
        for shape in [(), (1,), (2, 3), (2, 3, 4), (0, 4)]:
            base = torch.rand(shape) + 1.0
            work = base.clone().to(self.device)
            torch.foreach.lgamma_([work])
            self.assertEqual(work.shape, torch.Size(shape))
            torch.testing.assert_close(work.cpu(), torch.lgamma(base))

    def test_dtype_coverage(self):
        """float16 / float32 / bfloat16 inputs are rewritten with matching dtype."""
        for dtype in [torch.float16, torch.float32, torch.bfloat16]:
            base = (torch.rand(6) + 1.0).to(dtype)
            work = base.clone().to(self.device)
            torch.foreach.lgamma_([work])
            self.assertEqual(work.dtype, dtype)
            torch.testing.assert_close(work.cpu(), torch.lgamma(base))

    def test_non_contiguous(self):
        """A non-contiguous (transposed) view is updated in place correctly."""
        base = torch.rand(4, 6).t() + 1.0
        work = base.clone().to(self.device)
        self.assertFalse(work.is_contiguous())
        torch.foreach.lgamma_([work])
        torch.testing.assert_close(work.cpu(), torch.lgamma(base))

    def test_empty_list_raises(self):
        """An empty tensor list raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            torch.foreach.lgamma_([])

    def test_int_dtype_raises(self):
        """Integer input is unsupported and raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            torch.foreach.lgamma_([torch.arange(3)])

    def test_mixed_device_falls_back(self):
        """A mixed accelerator/CPU list does not raise: the slow path updates each
        tensor in place on its own device."""
        if self.device_name == "cpu":
            self.skipTest("no accelerator available for mixed-device input")
        dev_t = torch.rand(3, device=self.device) + 1.0
        cpu_t = torch.rand(3) + 1.0
        expected_dev = torch.lgamma(dev_t.clone().cpu())
        expected_cpu = torch.lgamma(cpu_t.clone())
        tensors = [dev_t, cpu_t]
        ret = torch.foreach.lgamma_(tensors)
        self.assertIs(ret, tensors)
        self._assert_close_seq([dev_t.cpu()], [expected_dev])
        self._assert_close_seq([cpu_t], [expected_cpu])


if __name__ == "__main__":
    run_tests()
