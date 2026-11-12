# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.swa_utils.get_ema_avg_fn 接口功能正确性
API 名称：torch.optim.swa_utils.get_ema_avg_fn
API 签名：get_ema_avg_fn(decay: float = 0.999) -> Callable[[Tensor, Tensor, Any], Tensor]

闭包真实公式（源码 torch/optim/swa_utils.py，已实测）：
    ema_update(ema, current, n) = decay * ema + (1 - decay) * current
num_averaged（第三参）被完全忽略；decay 越界（<0 或 >1）在工厂调用期抛 ValueError。
实测签名：本版本 get_ema_avg_fn 仅有 decay 一个参数（默认 0.999），无 multi 开关；
多模型变体是独立工厂 get_ema_multi_avg_fn(decay=0.999)。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 工厂返回 EMA 闭包；闭包对非空张量计算指数滑动平均 |
已覆盖（returns_callable/avg_fn_computes_weighted_average/decay_09_numeric_formula） |
| 枚举选项 | decay 取值分支：0.9 / 0.5 / 默认 0.999 |
已覆盖（decay_09_numeric_formula/decay_05_numeric_formula/default_decay_0999） |
| 参数类型 | decay 为 float；闭包三参 (ema: Tensor, current: Tensor, n: Any) 返回新 Tensor |
已覆盖（returns_new_tensor_not_inplace/num_averaged_ignored） |
| 传参与不传参 | 工厂不传 decay（默认 0.999）与显式传 decay 两条路径 | 已覆盖（test_default_decay_0999/test_decay_09_numeric_formula） |
| 等价类/边界值 | decay 边界 0/1 未逐一测值语义，仅测越界报错；n 任意取值不影响结果 | 已覆盖（test_num_averaged_ignored/test_decay_out_of_range_raises_value_error） |
| 正常传参场景 | ema = decay*ema + (1-decay)*current 手算比对；多模型变体与逐参数单版一致 |
已覆盖（decay_09_numeric_formula/decay_05_numeric_formula/multi_variant_relationship） |
| 异常传参场景 | decay 越界 ValueError（工厂期）；工厂多传参 TypeError；闭包缺参 TypeError |
已覆盖（decay_out_of_range_raises_value_error/no_arg_type_check/missing_arg_raises_type_error） |
| 混合设备类型 | 单对张量输入，NPU 路径与 CPU Golden 一致 | 已覆盖（test_npu_device_path） |

未覆盖项及原因：
- decay=0/1 的值语义：闭包为纯线性插值，0/1 端点退化为拷贝/恒等，无独立分支代码，仅保留越界报错断言。
- 实测无 multi 参数（本版本签名仅 decay）；多模型能力由独立工厂 get_ema_multi_avg_fn 提供，已在 test_multi_variant_relationship 中作关系断言。
- 本接口本质为框架状态工具，涉及浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。
"""
import inspect
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
class TestGet_ema_avg_fnget_ema_avg_fn(TestCase):
    """Test cases for torch.optim.swa_utils.get_ema_avg_fn (NPU/CUDA portable).

    The factory returns ema_update(ema, current, n) = decay * ema +
    (1 - decay) * current; num_averaged is ignored. decay outside [0, 1]
    raises ValueError at factory call time. This torch version exposes no
    `multi` flag: the multi-model variant is the separate factory
    get_ema_multi_avg_fn(decay).
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_returns_callable(self):
        """get_*_avg_fn returns a callable average function."""
        avg_fn = torch.optim.swa_utils.get_ema_avg_fn(decay=0.9)
        self.assertTrue(callable(avg_fn))

    def test_avg_fn_computes_weighted_average(self):
        """The returned fn averages an old average with a new model param."""
        avg_fn = torch.optim.swa_utils.get_ema_avg_fn(decay=0.9)
        avg = torch.empty(2)
        new = torch.ones(2)
        out = avg_fn(avg, new, torch.tensor(1))
        self.assertEqual(out.shape, avg.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_no_arg_type_check(self):
        """Passing unexpected positional args raises TypeError."""
        with self.assertRaises(TypeError):
            torch.optim.swa_utils.get_ema_avg_fn(1, 2, 3)

    def test_missing_arg_raises_type_error(self):
        """The closure requires all three positional args."""
        avg_fn = torch.optim.swa_utils.get_ema_avg_fn(decay=0.9)
        with self.assertRaises(TypeError):
            avg_fn(torch.zeros(2), torch.ones(2))

    def test_decay_09_numeric_formula(self):
        """decay=0.9: ema' = 0.9 * ema + 0.1 * current, chained twice."""
        avg_fn = torch.optim.swa_utils.get_ema_avg_fn(decay=0.9)
        ema = torch.tensor([1.0, 2.0])
        cur = torch.tensor([3.0, 4.0])
        out = avg_fn(ema, cur, 7)
        # 0.9*1 + 0.1*3 = 1.2 ; 0.9*2 + 0.1*4 = 2.2
        torch.testing.assert_close(out, torch.tensor([1.2, 2.2]))
        # Chain a second update: 0.9*1.2 + 0.1*5 = 1.58 ; 0.9*2.2 + 0.1*5 = 2.48
        out2 = avg_fn(out, torch.tensor([5.0, 5.0]), 8)
        torch.testing.assert_close(out2, torch.tensor([1.58, 2.48]))

    def test_decay_05_numeric_formula(self):
        """decay=0.5 gives the plain midpoint of ema and current."""
        avg_fn = torch.optim.swa_utils.get_ema_avg_fn(decay=0.5)
        ema = torch.tensor([1.0, 2.0])
        cur = torch.tensor([3.0, 4.0])
        out = avg_fn(ema, cur, 0)
        torch.testing.assert_close(out, torch.tensor([2.0, 3.0]))

    def test_default_decay_0999(self):
        """Omitting decay uses the documented default 0.999."""
        sig = inspect.signature(torch.optim.swa_utils.get_ema_avg_fn)
        self.assertEqual(list(sig.parameters), ['decay'],
                         "this torch version exposes only the decay parameter")
        self.assertEqual(sig.parameters['decay'].default, 0.999)
        avg_fn = torch.optim.swa_utils.get_ema_avg_fn()
        out = avg_fn(torch.tensor([1.0]), torch.tensor([3.0]), 0)
        # 0.999*1 + 0.001*3 = 1.002
        torch.testing.assert_close(out, torch.tensor([1.002]))

    def test_num_averaged_ignored(self):
        """The third argument is accepted but has no effect on the result."""
        avg_fn = torch.optim.swa_utils.get_ema_avg_fn(decay=0.9)
        ema = torch.tensor([1.0, 2.0])
        cur = torch.tensor([3.0, 4.0])
        outs = [avg_fn(ema, cur, n) for n in (0, 1, 5, 100)]
        for other in outs[1:]:
            torch.testing.assert_close(other, outs[0])

    def test_decay_out_of_range_raises_value_error(self):
        """decay < 0 or decay > 1 raises ValueError at factory call time."""
        with self.assertRaises(ValueError):
            torch.optim.swa_utils.get_ema_avg_fn(decay=1.5)
        with self.assertRaises(ValueError):
            torch.optim.swa_utils.get_ema_avg_fn(decay=-0.1)

    def test_returns_new_tensor_not_inplace(self):
        """The closure returns a fresh tensor and leaves inputs untouched."""
        avg_fn = torch.optim.swa_utils.get_ema_avg_fn(decay=0.9)
        ema = torch.tensor([1.0, 2.0])
        cur = torch.tensor([3.0, 4.0])
        out = avg_fn(ema, cur, 0)
        self.assertIsNot(out, ema, "closure must return a new tensor")
        self.assertIsNot(out, cur)
        torch.testing.assert_close(ema, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(cur, torch.tensor([3.0, 4.0]))

    def test_no_grad_closure(self):
        """The closure is decorated with @torch.no_grad: outputs carry no grad."""
        avg_fn = torch.optim.swa_utils.get_ema_avg_fn(decay=0.9)
        ema = torch.randn(3, requires_grad=True)
        cur = torch.randn(3, requires_grad=True)
        out = avg_fn(ema, cur, 1)
        self.assertFalse(out.requires_grad,
                         "ema_update runs under no_grad; output must not require grad")

    def test_multi_variant_relationship(self):
        """get_ema_multi_avg_fn(decay) equals per-param use of the single fn.

        Empirically this torch version has no `multi` flag on get_ema_avg_fn;
        the multi-model EMA is the separate factory get_ema_multi_avg_fn
        whose in-place foreach update must agree with the single closure
        applied to each parameter independently.
        """
        single_fn = torch.optim.swa_utils.get_ema_avg_fn(decay=0.9)
        multi_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9)
        emas = [torch.tensor([1.0, 2.0]), torch.tensor([10.0, 20.0])]
        curs = [torch.tensor([3.0, 4.0]), torch.tensor([30.0, 40.0])]
        expected = [single_fn(e, c, 0) for e, c in zip(emas, curs)]
        ret = multi_fn(emas, curs, 0)
        self.assertIsNone(ret, "the multi variant updates in place and returns None")
        for got, exp in zip(emas, expected):
            torch.testing.assert_close(got, exp)

    def test_npu_device_path(self):
        """The closure runs on device tensors and matches the CPU golden."""
        avg_fn = torch.optim.swa_utils.get_ema_avg_fn(decay=0.9)
        ema_cpu = torch.tensor([1.0, 2.0])
        cur_cpu = torch.tensor([3.0, 4.0])
        cpu_out = avg_fn(ema_cpu, cur_cpu, 3)
        dev_out = avg_fn(ema_cpu.to(self.device), cur_cpu.to(self.device), 3)
        torch.testing.assert_close(dev_out.cpu(), cpu_out)
        # chained second update on device still tracks the CPU golden
        cpu_out2 = avg_fn(cpu_out, torch.tensor([5.0, 5.0]), 4)
        dev_out2 = avg_fn(dev_out, torch.full((2,), 5.0, device=self.device), 4)
        torch.testing.assert_close(dev_out2.cpu(), cpu_out2)


if __name__ == "__main__":
    run_tests()
