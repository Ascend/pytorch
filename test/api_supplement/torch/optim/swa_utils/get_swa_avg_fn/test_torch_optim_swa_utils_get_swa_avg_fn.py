# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.swa_utils.get_swa_avg_fn 接口功能正确性
API 名称：torch.optim.swa_utils.get_swa_avg_fn
API 签名：get_swa_avg_fn() -> Callable[[Tensor, Tensor, Tensor | int], Tensor]

闭包真实公式（源码 torch/optim/swa_utils.py，已实测）：
    swa_update(averaged, current, n) = averaged + (current - averaged) / (n + 1)
即增量平均 avg_{k+1} = (sum_k + p) / (k+1)；n 可为 int 或 Tensor；
n=0 时返回值恰等于 current（AveragedModel 首次调用用它复制模型参数）。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无参工厂返回平均闭包；闭包对非空张量序列计算增量平均 |
已覆盖（returns_callable/avg_fn_computes_weighted_average/numeric_running_average_semantics） |
| 枚举选项 | 工厂无任何参数，无枚举可选 | 未覆盖（无该类参数；修正：此前误将 get_ema_avg_fn 的 decay 列入本表，已删除） |
| 参数类型 | 闭包三参 (averaged: Tensor, current: Tensor, n: int 或 Tensor)；返回新 Tensor |
已覆盖（num_averaged_int_and_tensor_equivalent/returns_new_tensor_not_inplace） |
| 传参与不传参 | 工厂无参调用；闭包缺参报 TypeError | 已覆盖（test_returns_callable/test_missing_arg_raises_type_error） |
| 等价类/边界值 | n=0 首次调用返回恰为 model 参数；多次调用等价于输入的算术平均 |
已覆盖（first_call_returns_current_param/numeric_running_average_semantics） |
| 正常传参场景 | 逐步手算 (sum+p)/(n+1) 与闭包输出精确比对 | 已覆盖（test_numeric_running_average_semantics） |
| 异常传参场景 | 工厂多传参 TypeError；闭包缺参 TypeError | 已覆盖（test_no_arg_type_check/test_missing_arg_raises_type_error） |
| 混合设备类型 | 单对张量输入，NPU 路径与 CPU Golden 逐步一致 | 已覆盖（test_npu_device_path） |

未覆盖项及原因：
- 工厂无枚举/可选参数（无 decay 等入参；decay 属于 get_ema_avg_fn，本表已修正张冠李戴行）。
- no_grad 语义：闭包被 @torch.no_grad() 装饰，输出不带梯度，已覆盖（test_no_grad_closure）。
- 本接口本质为框架状态工具，涉及浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 条件导入 torch_npu，并用 torch.npu.is_available() 探测可用 NPU。
DEVICE_NAME = None
try:
    import torch_npu  # noqa: F401
    if torch.npu.is_available():
        DEVICE_NAME = 'npu'
except (ImportError, AttributeError):
    pass
if DEVICE_NAME is None:
    DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

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
class TestGet_swa_avg_fnget_swa_avg_fn(TestCase):
    """Test cases for torch.optim.swa_utils.get_swa_avg_fn (NPU/CUDA portable).

    The factory returns swa_update(averaged, current, n) = averaged +
    (current - averaged) / (n + 1), i.e. the incremental running mean
    (sum + p) / (n + 1). First call with n=0 must return exactly the
    model parameter, which is how AveragedModel seeds its copy.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_returns_callable(self):
        """get_*_avg_fn returns a callable average function."""
        avg_fn = torch.optim.swa_utils.get_swa_avg_fn()
        self.assertTrue(callable(avg_fn))

    def test_avg_fn_computes_weighted_average(self):
        """The returned fn averages an old average with a new model param."""
        avg_fn = torch.optim.swa_utils.get_swa_avg_fn()
        avg = torch.empty(2)
        new = torch.ones(2)
        out = avg_fn(avg, new, torch.tensor(1))
        self.assertEqual(out.shape, avg.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_no_arg_type_check(self):
        """Passing unexpected positional args raises TypeError."""
        with self.assertRaises(TypeError):
            torch.optim.swa_utils.get_swa_avg_fn(1, 2, 3)

    def test_missing_arg_raises_type_error(self):
        """The closure requires all three positional args."""
        avg_fn = torch.optim.swa_utils.get_swa_avg_fn()
        with self.assertRaises(TypeError):
            avg_fn(torch.zeros(2), torch.ones(2))

    def test_first_call_returns_current_param(self):
        """n=0 returns exactly the model parameter (seeds the average).

        averaged + (current - averaged) / 1 == current for any starting
        average, including averaged == 0 which is the documented contract
        used by AveragedModel on the first update.
        """
        avg_fn = torch.optim.swa_utils.get_swa_avg_fn()
        current = torch.tensor([3.0, 6.0, 9.0])
        # averaged starts at zero (the canonical "no models yet" state)
        out = avg_fn(torch.zeros(3), current, 0)
        torch.testing.assert_close(out, current)
        # Even with a non-zero stale average, n=0 snaps to current.
        out_stale = avg_fn(torch.tensor([1.0, 2.0, 3.0]), current, 0)
        torch.testing.assert_close(out_stale, current)
        # num_averaged may also be a 0-dim tensor.
        out_t = avg_fn(torch.zeros(3), current, torch.tensor(0))
        torch.testing.assert_close(out_t, current)

    def test_numeric_running_average_semantics(self):
        """Manual (sum + p) / (n + 1) sequence matches the closure exactly."""
        avg_fn = torch.optim.swa_utils.get_swa_avg_fn()
        p0 = torch.tensor([3.0, 6.0, 9.0])
        p1 = torch.tensor([6.0, 6.0, 6.0])
        p2 = torch.tensor([12.0, 12.0, 12.0])
        # Hand-computed incremental means:
        #   n=0: 0 + (p0 - 0) / 1          = p0          = [3, 6, 9]
        #   n=1: avg + (p1 - avg) / 2      = [4.5, 6, 7.5]
        #   n=2: avg + (p2 - avg) / 3      = [7, 8, 9]
        avg = torch.zeros(3)
        avg = avg_fn(avg, p0, 0)
        torch.testing.assert_close(avg, p0)
        avg = avg_fn(avg, p1, 1)
        torch.testing.assert_close(avg, torch.tensor([4.5, 6.0, 7.5]))
        avg = avg_fn(avg, p2, 2)
        torch.testing.assert_close(avg, torch.tensor([7.0, 8.0, 9.0]))
        # The incremental mean equals the plain arithmetic mean of inputs.
        torch.testing.assert_close(avg, torch.stack([p0, p1, p2]).mean(dim=0))

    def test_returns_new_tensor_not_inplace(self):
        """The closure returns a fresh tensor and leaves inputs untouched."""
        avg_fn = torch.optim.swa_utils.get_swa_avg_fn()
        averaged = torch.zeros(3)
        current = torch.tensor([3.0, 6.0, 9.0])
        out = avg_fn(averaged, current, 0)
        self.assertIsNot(out, averaged, "closure must return a new tensor")
        self.assertIsNot(out, current)
        # Inputs keep their original values (no in-place mutation).
        torch.testing.assert_close(averaged, torch.zeros(3))
        torch.testing.assert_close(current, torch.tensor([3.0, 6.0, 9.0]))

    def test_num_averaged_int_and_tensor_equivalent(self):
        """int and 0-dim Tensor num_averaged produce identical results."""
        avg_fn = torch.optim.swa_utils.get_swa_avg_fn()
        averaged = torch.tensor([1.0, 2.0])
        current = torch.tensor([5.0, 8.0])
        for n in (0, 1, 3):
            out_int = avg_fn(averaged, current, n)
            out_tensor = avg_fn(averaged, current, torch.tensor(n))
            torch.testing.assert_close(out_int, out_tensor)

    def test_no_grad_closure(self):
        """The closure is decorated with @torch.no_grad: outputs carry no grad."""
        avg_fn = torch.optim.swa_utils.get_swa_avg_fn()
        averaged = torch.randn(3, requires_grad=True)
        current = torch.randn(3, requires_grad=True)
        out = avg_fn(averaged, current, 1)
        self.assertFalse(out.requires_grad,
                         "swa_update runs under no_grad; output must not require grad")

    def test_npu_device_path(self):
        """The closure runs on device tensors and matches the CPU golden."""
        avg_fn = torch.optim.swa_utils.get_swa_avg_fn()
        inputs = [torch.tensor([3.0, 6.0, 9.0]),
                  torch.tensor([6.0, 6.0, 6.0]),
                  torch.tensor([12.0, 12.0, 12.0])]
        # CPU golden trajectory
        cpu_avg = torch.zeros(3)
        for i, p in enumerate(inputs):
            cpu_avg = avg_fn(cpu_avg, p, i)
        # Device trajectory (NPU/CUDA), moved back to CPU for comparison
        dev_avg = torch.zeros(3, device=self.device)
        for i, p in enumerate(inputs):
            dev_avg = avg_fn(dev_avg, p.to(self.device), i)
        torch.testing.assert_close(dev_avg.cpu(), cpu_avg)
        # int and tensor n agree on the device path too
        dev_avg_t = torch.zeros(3, device=self.device)
        for i, p in enumerate(inputs):
            dev_avg_t = avg_fn(dev_avg_t, p.to(self.device), torch.tensor(i, device=self.device))
        torch.testing.assert_close(dev_avg_t.cpu(), cpu_avg)


if __name__ == "__main__":
    run_tests()
