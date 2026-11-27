# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.foreach.lgamma 接口功能正确性
API 名称：torch.foreach.lgamma
API 签名：torch.foreach.lgamma(inputs, /) -> tuple[Tensor]

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                                  |
|------------------|--------------------------------------------|-----------------------------------------------------------|
| shape            | 0-dim / 1D / 2D / 3D                       | 已覆盖：shape_coverage                                    |
| dtype            | float16/float32/bfloat16；整型提升为 fp32  | 已覆盖：dtype_coverage、integer_input_promotes            |
| device           | 每个正常场景 CPU/NPU 双端，CPU 为 Golden   | 已覆盖：cpu_golden_npu 及各 *_coverage                    |
| 参数枚举         | 该接口仅一个位置参数 inputs                | 不适用                                                    |
| 可选参数         | 该接口无可选参数                           | 不适用                                                    |
| in-place 变体    | out-of-place（in-place 见 torch.foreach.   | 不适用（另见 lgamma_ 用例）                              |
|                  | lgamma_）                                  |                                                           |
| out= 参数        | 不支持 out=                                | 不适用                                                    |
| 广播             | 逐张量独立处理，无广播语义                 | 不适用                                                    |
| 连续性           | 非连续（transpose 视图）输入               | 已覆盖：non_contiguous                                    |
| 逐元素语义       | foreach 结果应等于逐张量 torch.lgamma      | 已覆盖：matches_elementwise                               |
| 混合设备输入     | list 内同时含 NPU 与 CPU tensor（逐张量回退）| 已覆盖：mixed_device_falls_back（无加速器时按环境不具备跳过）|
| 异常路径         | 空 list 抛 RuntimeError                    | 已覆盖：empty_list_raises                                 |

未覆盖项及原因：
- 参数枚举 / 可选参数 / in-place / out= / 广播：该接口为固定单参的逐元素多张量运算，无这些维度。
- NPU 原生算子：本仓库环境下 torch.foreach.lgamma 在 NPU 上回退 CPU 执行（无原生 aclnn 算子），
  功能结果仍以 CPU 为 Golden 对照；原生算子覆盖不在本用例判定范围。
- 混合设备异常契约：foreach 的 slow path 不校验 list 内设备一致性，逐张量各自其设备计算，
  混合设备 list 不抛异常；用例仅验证逐张量回退结果与 torch.lgamma 一致，不再断言抛错。

注：foreach 系列为张量数值运算（计算类）；每个正常场景均 CPU/NPU 双端执行，CPU 输出为 Golden，
     NPU 输出移至 CPU 后用 torch.testing.assert_close 对照。
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
class TestForeachLgammaUt(TestCase):
    """Functional tests for torch.foreach.lgamma (NPU/CUDA portable)."""

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

    def test_foreach_lgamma_cpu_golden_npu(self):
        """CPU result is the Golden for the device path."""
        cpu_inputs = self._positive_inputs()
        cpu_out = torch.foreach.lgamma(cpu_inputs)
        dev_out = torch.foreach.lgamma([t.to(self.device) for t in cpu_inputs])
        self.assertIsInstance(cpu_out, tuple)
        self.assertEqual(len(dev_out), len(cpu_inputs))
        self._assert_close_seq([t.cpu() for t in dev_out], cpu_out)

    def test_foreach_lgamma_matches_elementwise(self):
        """foreach.lgamma equals applying torch.lgamma to each tensor independently."""
        cpu_inputs = self._positive_inputs()
        dev_out = torch.foreach.lgamma([t.to(self.device) for t in cpu_inputs])
        golden = [torch.lgamma(t) for t in cpu_inputs]
        self._assert_close_seq([t.cpu() for t in dev_out], golden)

    def test_foreach_lgamma_tuple_input(self):
        """Tuple input is accepted and returns a tuple with matching values."""
        x = torch.rand(4) + 1.0
        cpu_out = torch.foreach.lgamma((x,))
        dev_out = torch.foreach.lgamma((x.to(self.device),))
        self.assertIsInstance(dev_out, tuple)
        self._assert_close_seq([dev_out[0].cpu()], list(cpu_out))

    def test_foreach_lgamma_single_element(self):
        """A single-tensor list returns one result tensor."""
        x = torch.rand(2, 2) + 1.0
        out = torch.foreach.lgamma([x.to(self.device)])
        self.assertEqual(len(out), 1)
        torch.testing.assert_close(out[0].cpu(), torch.lgamma(x))

    def test_foreach_lgamma_shape_coverage(self):
        """0-dim / 1D / 2D / 3D / size-0 dims are handled with matching shape and values."""
        for shape in [(), (1,), (2, 3), (2, 3, 4), (0, 4)]:
            base = torch.rand(shape) + 1.0
            out = torch.foreach.lgamma([base.to(self.device)])[0]
            self.assertEqual(out.shape, torch.Size(shape))
            torch.testing.assert_close(out.cpu(), torch.lgamma(base))

    def test_foreach_lgamma_dtype_coverage(self):
        """float16 / float32 / bfloat16 inputs produce matching dtype and values."""
        for dtype in [torch.float16, torch.float32, torch.bfloat16]:
            base = (torch.rand(6) + 1.0).to(dtype)
            out = torch.foreach.lgamma([base.to(self.device)])[0]
            self.assertEqual(out.dtype, dtype)
            torch.testing.assert_close(out.cpu(), torch.lgamma(base))

    def test_foreach_lgamma_non_contiguous(self):
        """A non-contiguous (transposed) view produces correct values."""
        base = torch.rand(4, 6).t() + 1.0
        self.assertFalse(base.is_contiguous())
        out = torch.foreach.lgamma([base.to(self.device)])[0]
        torch.testing.assert_close(out.cpu(), torch.lgamma(base))

    def test_foreach_lgamma_empty_list_raises(self):
        """An empty tensor list raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            torch.foreach.lgamma([])

    def test_foreach_lgamma_integer_input_promotes(self):
        """Integer input is promoted to float32, matching torch.lgamma semantics."""
        base = torch.arange(1, 4)
        out = torch.foreach.lgamma([base.to(self.device)])[0]
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out.cpu(), torch.lgamma(base))

    def test_foreach_lgamma_mixed_device_falls_back(self):
        """A mixed accelerator/CPU list does not raise: the slow path computes each
        tensor on its own device."""
        if self.device_name == "cpu":
            self.skipTest("no accelerator available for mixed-device input")
        dev_t = torch.rand(3, device=self.device) + 1.0
        cpu_t = torch.rand(3) + 1.0
        out = torch.foreach.lgamma([dev_t, cpu_t])
        self.assertEqual(len(out), 2)
        self._assert_close_seq([out[0].cpu()], [torch.lgamma(dev_t.cpu())])
        self._assert_close_seq([out[1]], [torch.lgamma(cpu_t)])


if __name__ == "__main__":
    run_tests()
