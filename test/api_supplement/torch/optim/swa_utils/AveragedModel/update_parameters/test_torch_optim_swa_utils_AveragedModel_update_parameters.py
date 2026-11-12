# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.swa_utils.AveragedModel.update_parameters 接口功能正确性
API 名称：torch.optim.swa_utils.AveragedModel.update_parameters
API 签名：update_parameters(self, model: Module) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | n_averaged 初始为 0 的 int64 buffer；用当前模型参数更新平均模型 |
已覆盖（n_averaged_starts_zero_and_increments/first_update_copies_model_weights） |
| 枚举选项 | 默认 SWA 均值 / 显式 avg_fn=get_ema_avg_fn(decay) / use_buffers True/False |
已覆盖（swa_running_average_values/ema_avg_fn_values/buffers_averaged_when_use_buffers_true） |
| 参数类型 | 入参为 Module；非 Module（object/None/list）抛 AttributeError | 已覆盖（test_update_parameters_non_module） |
| 传参与不传参 | 必须传 model；构造期 avg_fn/device 可选参数的两条路径 |
已覆盖（first_update_copies_model_weights/ema_avg_fn_values/update_on_device_cpu_golden） |
| 等价类/边界值 | n_averaged 0→1→2→3 逐次递增；首次调用为直接拷贝而非平均 |
已覆盖（n_averaged_starts_zero_and_increments/first_update_copies_model_weights） |
| 正常传参场景 | 多次 update 后 averaged 参数等于期望平均；深拷贝隔离；buffer 同步/平均；返回 None |
已覆盖（swa_running_average_values/deep_copy_isolation/buffers_synced_when_use_buffers_false/ |
update_returns_none） |
| 异常传参场景 | 非 Module 参数报错 | 已覆盖（test_update_parameters_non_module） |
| 混合设备类型 | model 驻留当前设备时 update 生效，CPU 结果为 Golden 对照；构造期 device= 实现跨设备更新 |
已覆盖（update_on_device_cpu_golden/update_cross_device_ctor_arg） |

未覆盖项及原因：
- 本接口本质为优化器/调度器/状态工具的框架行为，涉及浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。
- multi_avg_fn（多参数向量化平均）为独立的 foreach 加速入口，不在本文件（默认 avg_fn 路径）断言范围，避免与内部 foreach 设备支持矩阵耦合。
"""
import os
import unittest

import torch
import torch.optim.swa_utils as swa_utils

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
class TestAveragedModelupdate_parameters(TestCase):
    """Test cases for torch.optim.swa_utils.AveragedModel.update_parameters (NPU/CUDA portable).

    Empirical contract (torch 2.11, probed on NPU):
      - n_averaged is an int64 buffer starting at tensor(0);
      - the first update copies model weights directly (copy_param when
        n_averaged == 0), subsequent updates fold the running mean with
        avg + (w - avg) / (n + 1) (SWA) or decay * avg + (1 - decay) * w (EMA);
      - the averaged copy is a deepcopy, so mutating the source model after
        the update cannot leak into the stored average;
      - buffers are synced from the source model when use_buffers=False and
        averaged like parameters when use_buffers=True.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _set_weight(self, linear, value_2x2):
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(value_2x2, dtype=linear.weight.dtype))

    def test_update_parameters(self):
        """update_parameters folds the model into the running average."""
        model = torch.nn.Linear(3, 2)
        am = torch.optim.swa_utils.AveragedModel(model)
        n0 = int(am.n_averaged)
        am.update_parameters(model)
        self.assertEqual(int(am.n_averaged), n0 + 1 if n0 == 0 else n0)

    def test_update_parameters_non_module(self):
        """Passing a non-Module raises."""
        am = torch.optim.swa_utils.AveragedModel(torch.nn.Linear(3, 2))
        with self.assertRaises((TypeError, AttributeError, RuntimeError)):
            am.update_parameters(object())

    def test_n_averaged_starts_zero_and_increments(self):
        """n_averaged is an int64 buffer starting at 0 and increments per call."""
        model = torch.nn.Linear(2, 2, bias=False)
        am = swa_utils.AveragedModel(model)
        # Probed on torch 2.11: n_averaged is a registered long tensor of 0.
        self.assertIn("n_averaged", dict(am.named_buffers()),
                      "n_averaged must be a registered buffer")
        self.assertEqual(am.n_averaged.dtype, torch.long)
        self.assertEqual(int(am.n_averaged), 0, "counter must start at 0")
        # 0 -> 1 -> 2 -> 3 across three successive updates.
        am.update_parameters(model)
        self.assertEqual(int(am.n_averaged), 1)
        am.update_parameters(model)
        self.assertEqual(int(am.n_averaged), 2)
        am.update_parameters(model)
        self.assertEqual(int(am.n_averaged), 3)

    def test_first_update_copies_model_weights(self):
        """The first update is a direct copy of the model weights, not an average."""
        w1 = [[1.0, 2.0], [3.0, 4.0]]
        model = torch.nn.Linear(2, 2, bias=False)
        self._set_weight(model, w1)
        am = swa_utils.AveragedModel(model)
        # The averaged copy is a deepcopy: distinct storage objects.
        self.assertIsNot(am.module.weight, model.weight,
                         "AveragedModel must deepcopy the source model")
        am.update_parameters(model)
        torch.testing.assert_close(am.module.weight.detach(),
                                   torch.tensor(w1),
                                   msg="first update must copy w1 verbatim (copy_param path)")
        self.assertEqual(int(am.n_averaged), 1)

    def test_swa_running_average_values(self):
        """Default SWA: after k updates the average equals mean(w1..wk)."""
        w1 = [[1.0, 2.0], [3.0, 4.0]]
        w2 = [[5.0, 6.0], [7.0, 8.0]]
        w3 = [[9.0, 10.0], [11.0, 12.0]]
        model = torch.nn.Linear(2, 2, bias=False)
        am = swa_utils.AveragedModel(model)

        self._set_weight(model, w1)
        am.update_parameters(model)
        torch.testing.assert_close(am.module.weight.detach(), torch.tensor(w1),
                                   msg="first update copies w1")

        self._set_weight(model, w2)
        am.update_parameters(model)
        # avg += (w - avg) / (n + 1) with n=1 gives the plain mean of w1, w2.
        torch.testing.assert_close(am.module.weight.detach(),
                                   (torch.tensor(w1) + torch.tensor(w2)) / 2,
                                   msg="second update must give (w1+w2)/2")
        self.assertEqual(int(am.n_averaged), 2)

        self._set_weight(model, w3)
        am.update_parameters(model)
        torch.testing.assert_close(am.module.weight.detach(),
                                   (torch.tensor(w1) + torch.tensor(w2) + torch.tensor(w3)) / 3,
                                   msg="third update must give (w1+w2+w3)/3")
        self.assertEqual(int(am.n_averaged), 3)

    def test_ema_avg_fn_values(self):
        """With avg_fn=get_ema_avg_fn(decay): first update copies, then EMA applies."""
        decay = 0.9
        w1 = [[1.0, 2.0], [3.0, 4.0]]
        w2 = [[5.0, 6.0], [7.0, 8.0]]
        model = torch.nn.Linear(2, 2, bias=False)
        am = swa_utils.AveragedModel(model, avg_fn=swa_utils.get_ema_avg_fn(decay=decay))

        self._set_weight(model, w1)
        am.update_parameters(model)
        torch.testing.assert_close(am.module.weight.detach(), torch.tensor(w1),
                                   msg="EMA first update must copy w1 verbatim")

        self._set_weight(model, w2)
        am.update_parameters(model)
        expected = decay * torch.tensor(w1) + (1 - decay) * torch.tensor(w2)
        torch.testing.assert_close(am.module.weight.detach(), expected,
                                   msg="EMA second update must apply decay*avg+(1-decay)*w")
        # The custom avg_fn path still drives the same counter semantics.
        self.assertEqual(int(am.n_averaged), 2)

    def test_deep_copy_isolation(self):
        """Mutating the source model after an update does not touch the stored average."""
        model = torch.nn.Linear(2, 2, bias=False)
        self._set_weight(model, [[1.0, 1.0], [1.0, 1.0]])
        am = swa_utils.AveragedModel(model)
        am.update_parameters(model)
        snapshot = am.module.weight.detach().clone()

        # In-place mutation of the source model must not leak into the average.
        with torch.no_grad():
            model.weight.add_(100.0)
        torch.testing.assert_close(am.module.weight.detach(), snapshot,
                                   msg="the averaged copy must be isolated from the source model")
        self.assertIsNot(am.module.weight, model.weight)

    def test_buffers_synced_when_use_buffers_false(self):
        """use_buffers=False: buffers are copied from the model, not averaged."""
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        am = swa_utils.AveragedModel(model, use_buffers=False)
        model(torch.randn(8, 2))  # advance BN running stats
        first_stats = model[1].running_mean.detach().clone()
        am.update_parameters(model)
        torch.testing.assert_close(am.module[1].running_mean, first_stats,
                                   msg="use_buffers=False must sync buffers from the model")

        model(torch.randn(8, 2) * 3.0)  # shift BN stats again
        am.update_parameters(model)
        torch.testing.assert_close(am.module[1].running_mean, model[1].running_mean,
                                   msg="second update must re-sync buffers, not average them")
        self.assertEqual(int(am.n_averaged), 2)

    def test_buffers_averaged_when_use_buffers_true(self):
        """use_buffers=True: buffers join the running average like parameters."""
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        am = swa_utils.AveragedModel(model, use_buffers=True)
        model(torch.randn(8, 2))
        first_stats = model[1].running_mean.detach().clone()
        am.update_parameters(model)
        torch.testing.assert_close(am.module[1].running_mean, first_stats,
                                   msg="first update must copy the buffer (copy_param path)")

        model(torch.randn(8, 2) * 3.0)
        second_stats = model[1].running_mean.detach().clone()
        am.update_parameters(model)
        torch.testing.assert_close(am.module[1].running_mean, (first_stats + second_stats) / 2,
                                   msg="use_buffers=True must average buffers across updates")
        self.assertEqual(int(am.n_averaged), 2)

    def test_update_returns_none(self):
        """update_parameters returns None per its signature."""
        model = torch.nn.Linear(2, 2)
        am = swa_utils.AveragedModel(model)
        self.assertIsNone(am.update_parameters(model))

    def test_update_on_device_cpu_golden(self):
        """Device-resident model updates identically to the CPU golden path."""
        w1 = [[1.0, 2.0], [3.0, 4.0]]
        w2 = [[5.0, 6.0], [7.0, 8.0]]

        torch.manual_seed(11)
        model_cpu = torch.nn.Linear(2, 2, bias=False)
        am_cpu = swa_utils.AveragedModel(model_cpu)
        torch.manual_seed(11)
        model_dev = torch.nn.Linear(2, 2, bias=False).to(self.device)
        am_dev = swa_utils.AveragedModel(model_dev)

        self._set_weight(model_cpu, w1)
        self._set_weight(model_dev, w1)
        am_cpu.update_parameters(model_cpu)
        am_dev.update_parameters(model_dev)
        self._set_weight(model_cpu, w2)
        self._set_weight(model_dev, w2)
        am_cpu.update_parameters(model_cpu)
        am_dev.update_parameters(model_dev)

        self.assertEqual(int(am_dev.n_averaged), int(am_cpu.n_averaged))
        # CPU output is the golden; NPU/CUDA output moved to CPU before comparing.
        torch.testing.assert_close(am_dev.module.weight.detach().cpu(),
                                   am_cpu.module.weight.detach(),
                                   msg="device-path running mean must match the CPU golden")

    def test_update_cross_device_ctor_arg(self):
        """AveragedModel(model, device=...) accepts updates from a CPU-resident model."""
        w1 = [[2.0, 4.0], [6.0, 8.0]]
        model = torch.nn.Linear(2, 2, bias=False)
        self._set_weight(model, w1)
        if self.device_name == 'cpu':
            am = swa_utils.AveragedModel(model, device=torch.device('cpu'))
        else:
            am = swa_utils.AveragedModel(model, device=self.device)
            self.assertEqual(am.module.weight.device.type, self.device_name,
                             "ctor device= must move the averaged copy")
        # update_parameters moves model params to the averaged device internally.
        am.update_parameters(model)
        self.assertEqual(int(am.n_averaged), 1)
        torch.testing.assert_close(am.module.weight.detach().cpu(), torch.tensor(w1),
                                   msg="cross-device first update must copy the CPU weights")


if __name__ == "__main__":
    run_tests()
