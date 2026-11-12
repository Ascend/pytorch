# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.swa_utils.get_swa_multi_avg_fn 接口功能正确性
API 名称：torch.optim.swa_utils.get_swa_multi_avg_fn
API 签名：get_swa_multi_avg_fn() -> Callable[[list[Tensor], list[Tensor], Tensor | int], None]

闭包真实公式（源码 torch/optim/swa_utils.py，已实测）：
    swa_update(averaged_list, current_list, n) 原地更新（返回 None）：
    浮点分支 torch._foreach_lerp_(averaged, current, 1/(n+1))
      即 averaged += (current - averaged) / (n + 1) —— 每个模型独立增量平均；
    整数分支（非浮点/复数张量）在 torch 2.11 因 addcdiv 整数除法移除与
    _foreach_add_ 浮点 alpha 限制直接 RuntimeError，属上游限制（见未覆盖项）。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无参工厂返回多模型平均闭包；对两个模型的非空张量列表计算增量平均 |
已覆盖（returns_callable/avg_fn_computes_weighted_average/two_models_independent_numeric_sequence） |
| 枚举选项 | 工厂无任何参数，无枚举可选 | 未覆盖（无该类参数；修正：此前误将 get_ema_avg_fn 的 decay 列入本表，已删除） |
| 参数类型 | 闭包三参 (list[Tensor], list[Tensor], int 或 Tensor)；原地更新返回 None |
已覆盖（inplace_update_returns_none/tensor_num_averaged_path） |
| 传参与不传参 | 工厂无参调用；闭包缺参报 TypeError | 已覆盖（test_returns_callable/test_missing_arg_raises_type_error） |
| 等价类/边界值 | n=0 首次调用各模型平均恰等于 current；两模型数值量级不同互不影响 | 已覆盖（test_two_models_independent_numeric_sequence） |
| 正常传参场景 | 各模型独立 (sum_i + p_i)/(n+1) 手算比对；multi 版与逐模型跑单版 get_swa_avg_fn 数值一致 |
已覆盖（two_models_independent_numeric_sequence/equivalent_to_per_model_single_fn） |
| 异常传参场景 | 工厂多传参 TypeError；闭包缺参 TypeError | 已覆盖（test_no_arg_type_check/test_missing_arg_raises_type_error） |
| 混合设备类型 | 设备张量列表路径与 CPU Golden 一致（含 tensor 型 n） | 已覆盖（test_npu_device_path） |

未覆盖项及原因：
- 工厂无枚举/可选参数（无 decay 等入参；decay 属于 get_ema_avg_fn/get_ema_multi_avg_fn，本表已修正张冠李戴行）。
- 整数 dtype 分支：averaged 为非浮点张量时，torch 2.11 上 tensor-n 走 _foreach_addcdiv_ 触发
  "Integer division with addcdiv is no longer supported"RuntimeError；int-n 走 _foreach_add_ 触发
  "alpha must not be a floating point number"RuntimeError；两条路径均为上游限制而非可测契约，不作为断言。
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
class TestGet_swa_multi_avg_fnget_swa_multi_avg_fn(TestCase):
    """Test cases for torch.optim.swa_utils.get_swa_multi_avg_fn (NPU/CUDA portable).

    The factory returns an in-place swa_update(averaged_list, current_list, n):
    each model's average evolves independently as averaged += (current -
    averaged) / (n + 1) (foreach lerp with weight 1/(n+1)), which must match
    running get_swa_avg_fn per model.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_returns_callable(self):
        """get_*_avg_fn returns a callable average function."""
        avg_fn = torch.optim.swa_utils.get_swa_multi_avg_fn()
        self.assertTrue(callable(avg_fn))

    def test_avg_fn_computes_weighted_average(self):
        """The returned fn averages an old average with a new model param."""
        avg_fn = torch.optim.swa_utils.get_swa_multi_avg_fn()
        avg = [torch.zeros(2)]
        new = [torch.ones(2)]
        avg_fn(avg, new, 1)
        # 1/(num_averaged+1) = 0.5 weight toward new
        torch.testing.assert_close(avg[0], torch.full((2,), 0.5))

    def test_no_arg_type_check(self):
        """Passing unexpected positional args raises TypeError."""
        with self.assertRaises(TypeError):
            torch.optim.swa_utils.get_swa_multi_avg_fn(1, 2, 3)

    def test_missing_arg_raises_type_error(self):
        """The closure requires all three positional args."""
        avg_fn = torch.optim.swa_utils.get_swa_multi_avg_fn()
        with self.assertRaises(TypeError):
            avg_fn([torch.zeros(2)], [torch.ones(2)])

    def test_inplace_update_returns_none(self):
        """The closure returns None and mutates the averaged list in place."""
        avg_fn = torch.optim.swa_utils.get_swa_multi_avg_fn()
        avg = [torch.zeros(2), torch.zeros(2)]
        cur = [torch.tensor([2.0, 4.0]), torch.tensor([10.0, 20.0])]
        ptrs = [t.data_ptr() for t in avg]
        ret = avg_fn(avg, cur, 0)
        self.assertIsNone(ret, "multi swa_update mutates in place and returns None")
        # Same storage, values updated: the caller's tensors were reused.
        self.assertEqual([t.data_ptr() for t in avg], ptrs)
        torch.testing.assert_close(avg[0], torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(avg[1], torch.tensor([10.0, 20.0]))

    def test_two_models_independent_numeric_sequence(self):
        """Two models evolve independently: (sum_i + p_i) / (n + 1) each."""
        avg_fn = torch.optim.swa_utils.get_swa_multi_avg_fn()
        # Deliberately different scales so cross-model interference would show.
        a1, a2 = torch.zeros(2), torch.zeros(2)
        # n=0: each average snaps to its own current value
        avg_fn([a1, a2], [torch.tensor([2.0, 4.0]), torch.tensor([10.0, 20.0])], 0)
        torch.testing.assert_close(a1, torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(a2, torch.tensor([10.0, 20.0]))
        # n=1: a1 += ([6,8] - a1)/2 -> [4,6]; a2 += ([30,50] - a2)/2 -> [20,35]
        avg_fn([a1, a2], [torch.tensor([6.0, 8.0]), torch.tensor([30.0, 50.0])], 1)
        torch.testing.assert_close(a1, torch.tensor([4.0, 6.0]))
        torch.testing.assert_close(a2, torch.tensor([20.0, 35.0]))
        # Cross-check against per-model arithmetic means of the inputs.
        torch.testing.assert_close(
            a1, torch.stack([torch.tensor([2.0, 4.0]), torch.tensor([6.0, 8.0])]).mean(dim=0))
        torch.testing.assert_close(
            a2, torch.stack([torch.tensor([10.0, 20.0]), torch.tensor([30.0, 50.0])]).mean(dim=0))
        # n=2 with a third model snapshot: a1 += ([4,4] - a1)/3 -> [4, 16/3]
        avg_fn([a1, a2], [torch.tensor([4.0, 4.0]), torch.tensor([20.0, 20.0])], 2)
        torch.testing.assert_close(a1, torch.tensor([4.0, 16.0 / 3.0]))
        torch.testing.assert_close(a2, torch.tensor([20.0, 30.0]))

    def test_tensor_num_averaged_path(self):
        """num_averaged as a 0-dim Tensor matches the int path."""
        avg_fn = torch.optim.swa_utils.get_swa_multi_avg_fn()
        avg_int = [torch.zeros(2)]
        avg_tensor = [torch.zeros(2)]
        cur = [torch.tensor([3.0, 6.0])]
        avg_fn(avg_int, cur, 1)
        avg_fn(avg_tensor, cur, torch.tensor(1))
        torch.testing.assert_close(avg_tensor[0], avg_int[0])

    def test_equivalent_to_per_model_single_fn(self):
        """Multi fn on a list equals get_swa_avg_fn applied per model."""
        multi_fn = torch.optim.swa_utils.get_swa_multi_avg_fn()
        single_fn = torch.optim.swa_utils.get_swa_avg_fn()
        model_snapshots = [
            (torch.tensor([2.0, 4.0]), torch.tensor([10.0, 20.0])),
            (torch.tensor([6.0, 8.0]), torch.tensor([30.0, 50.0])),
            (torch.tensor([4.0, 4.0]), torch.tensor([20.0, 20.0])),
        ]
        m_avg = [torch.zeros(2), torch.zeros(2)]
        s_avg = [torch.zeros(2), torch.zeros(2)]
        for n, (v1, v2) in enumerate(model_snapshots):
            multi_fn(m_avg, [v1, v2], n)
            s_avg[0] = single_fn(s_avg[0], v1, n)
            s_avg[1] = single_fn(s_avg[1], v2, n)
            # Track equality at every step, not only at the end.
            torch.testing.assert_close(m_avg[0], s_avg[0])
            torch.testing.assert_close(m_avg[1], s_avg[1])

    def test_npu_device_path(self):
        """Device-tensor lists track the CPU golden trajectory exactly."""
        avg_fn = torch.optim.swa_utils.get_swa_multi_avg_fn()
        snapshots = [
            (torch.tensor([2.0, 4.0]), torch.tensor([10.0, 20.0])),
            (torch.tensor([6.0, 8.0]), torch.tensor([30.0, 50.0])),
        ]
        cpu_avg = [torch.zeros(2), torch.zeros(2)]
        dev_avg = [torch.zeros(2, device=self.device), torch.zeros(2, device=self.device)]
        for n, (v1, v2) in enumerate(snapshots):
            avg_fn(cpu_avg, [v1, v2], n)
            avg_fn(dev_avg, [v1.to(self.device), v2.to(self.device)], n)
            torch.testing.assert_close(dev_avg[0].cpu(), cpu_avg[0])
            torch.testing.assert_close(dev_avg[1].cpu(), cpu_avg[1])
        # tensor num_averaged on the device path agrees too
        cpu_avg_t = [torch.zeros(2), torch.zeros(2)]
        dev_avg_t = [torch.zeros(2, device=self.device), torch.zeros(2, device=self.device)]
        for n, (v1, v2) in enumerate(snapshots):
            avg_fn(cpu_avg_t, [v1, v2], torch.tensor(n))
            avg_fn(dev_avg_t, [v1.to(self.device), v2.to(self.device)],
                   torch.tensor(n, device=self.device))
            torch.testing.assert_close(dev_avg_t[0].cpu(), cpu_avg_t[0])
            torch.testing.assert_close(dev_avg_t[1].cpu(), cpu_avg_t[1])


if __name__ == "__main__":
    run_tests()
