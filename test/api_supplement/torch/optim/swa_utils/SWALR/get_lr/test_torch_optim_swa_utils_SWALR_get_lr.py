# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.swa_utils.SWALR.get_lr 接口功能正确性
API 名称：torch.optim.swa_utils.SWALR.get_lr
API 签名：get_lr(self) -> list[float | Tensor]

退火公式（源码 torch/optim/swa_utils.py，已逐步实测，base_lr=0.1、swa_lr=0.05、
anneal_epochs=10 时第 e 次 step 后 get_last_lr() 即 get_lr 在步内的输出）：
- t = min(1, e / anneal_epochs)，e > anneal_epochs 后恒为 swa_lr；
- cos 分支：   alpha = (1 - cos(pi * t)) / 2，lr = swa_lr*alpha + base_lr*(1-alpha)；
- linear 分支：alpha = t，            lr = swa_lr*alpha + base_lr*(1-alpha)；
- anneal_epochs=0 边界：构造后即为 swa_lr（内部 step=max(1, step) 使 t=1）；
- get_lr() 在 step() 之外直接调用会发 UserWarning 提示改用 get_last_lr()，
  且按当前 group["lr"] 反推初值后再套公式，因此非步内语义以 get_last_lr 为准；
- SWALR 构造时把 swa_lr（标量或列表）写入各 param_groups[i]["swa_lr"]，
  多参数组各自独立退火。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无参调用返回与参数组数一致的学习率列表 | 已覆盖（test_get_lr_returns_list/test_multi_group_swa_lr_list） |
| 枚举选项 | anneal_strategy cos 与 linear 两分支数值断言（此前仅测默认 cos，已补真） |
已覆盖（cos_anneal_numeric_trajectory/linear_anneal_numeric_trajectory/cos_and_linear_branches_differ） |
| 参数类型 | 返回 list[float]；swa_lr 支持 float 与 list[float] | 已覆盖（test_get_lr_returns_list/test_multi_group_swa_lr_list） |
| 传参与不传参 | 无参调用；swa_lr 传标量或列表两条构造路径 | 已覆盖（test_swa_lr_written_to_param_groups/test_multi_group_swa_lr_list） |
| 等价类/边界值 | 初值=base_lrs；e=anneal_epochs 到达 swa_lr 并保持；anneal_epochs=0 立即 swa_lr |
已覆盖（initial_lr_matches_base/swa_lr_reached_and_held/anneal_epochs_zero_immediate_swa_lr） |
| 正常传参场景 | 两分支逐步与退火公式精确比对；swa_lr 生效断言 |
已覆盖（cos_anneal_numeric_trajectory/linear_anneal_numeric_trajectory/swa_lr_written_to_param_groups） |
| 异常传参场景 | get_lr 自身无异常路径；决定 get_lr 退火函数的构造参数非法时报 ValueError |
（anneal_strategy 非 cos/linear、anneal_epochs<0 或非 int） |
已覆盖（invalid_anneal_strategy_raises_value_error） |
| 混合设备类型 | 纯调度状态（host 侧标量计算），无张量迁移语义 | 未覆盖（无设备对照语义） |

未覆盖项及原因：
- get_lr 步外直接调用的返回值依赖当前 group["lr"] 的反推细节，属实现内部口径；仅断言其构造初值口径（=base lr）
  与 UserWarning 行为（test_direct_get_lr_warns_to_use_get_last_lr），轨迹断言以步内输出（get_last_lr）为准。
- 本接口为框架调度状态工具，lr 计算在 host 侧完成，无浮点张量的设备对照语义。
"""
import math
import os
import unittest
import warnings

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
class TestSWALRget_lr(TestCase):
    """Test cases for torch.optim.swa_utils.SWALR.get_lr (NPU/CUDA portable).

    get_lr interpolates each group from base_lr to swa_lr over
    anneal_epochs: cos uses alpha = (1 - cos(pi*t))/2, linear uses
    alpha = t, with t = min(1, e/anneal_epochs); past anneal_epochs the
    lr is pinned at swa_lr. With base_lr=0.1, swa_lr=0.05 and
    anneal_epochs=10 the two branches differ mid-anneal (e.g. e=3 gives
    ~0.08969 for cos vs 0.085 for linear) and converge to 0.05 at e=10.
    """

    BASE_LR = 0.1
    SWA_LR = 0.05
    ANNEAL_EPOCHS = 10

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    @staticmethod
    def _expected_alpha(strategy, epoch, anneal_epochs):
        t = min(1.0, epoch / anneal_epochs)
        if strategy == 'cos':
            return (1.0 - math.cos(math.pi * t)) / 2.0
        return t

    def _expected_lr(self, strategy, epoch, base_lr, swa_lr, anneal_epochs):
        alpha = self._expected_alpha(strategy, epoch, anneal_epochs)
        return swa_lr * alpha + base_lr * (1.0 - alpha)

    @staticmethod
    def _make_scheduler(strategy, swa_lr=0.05, anneal_epochs=10, lr=0.1):
        opt = torch.optim.SGD(
            [torch.nn.Parameter(torch.zeros(1), requires_grad=True)], lr=lr)
        sched = torch.optim.swa_utils.SWALR(
            opt, swa_lr=swa_lr, anneal_epochs=anneal_epochs,
            anneal_strategy=strategy)
        opt.step()  # mark optimizer called (avoids the step-order warning)
        return opt, sched

    def test_get_lr_returns_list(self):
        """SWALR.get_lr returns a per-group learning-rate list."""
        opt = torch.optim.SGD(
            [torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.1)
        swa = torch.optim.swa_utils.SWALR(opt, swa_lr=0.05)
        lr = swa.get_lr()
        self.assertIsInstance(lr, list)
        self.assertEqual(len(lr), 1)
        self.assertTrue(all(isinstance(x, float) for x in lr))

    def test_initial_lr_matches_base(self):
        """At construction the initial step keeps the lr at base_lrs."""
        opt, sched = self._make_scheduler('cos')
        self.assertEqual(sched.base_lrs, [self.BASE_LR])
        self.assertEqual(sched.get_last_lr(), [self.BASE_LR])
        self.assertEqual(opt.param_groups[0]['lr'], self.BASE_LR)

    def test_cos_anneal_numeric_trajectory(self):
        """cos branch matches alpha=(1-cos(pi*t))/2 exactly, then pins at swa_lr."""
        opt, sched = self._make_scheduler('cos')
        got = []
        for e in range(1, 13):
            sched.step()
            got.append(sched.get_last_lr()[0])
        expected = [self._expected_lr('cos', e, self.BASE_LR, self.SWA_LR,
                                      self.ANNEAL_EPOCHS) for e in range(1, 13)]
        torch.testing.assert_close(
            torch.tensor(got, dtype=torch.float64),
            torch.tensor(expected, dtype=torch.float64))
        # Past anneal_epochs the lr is pinned at swa_lr (e=11, 12 above).
        self.assertEqual(got[10], self.SWA_LR)
        self.assertEqual(got[11], self.SWA_LR)
        self.assertEqual(opt.param_groups[0]['lr'], self.SWA_LR)

    def test_linear_anneal_numeric_trajectory(self):
        """linear branch matches alpha=t exactly, then pins at swa_lr."""
        opt, sched = self._make_scheduler('linear')
        got = []
        for e in range(1, 13):
            sched.step()
            got.append(sched.get_last_lr()[0])
        expected = [self._expected_lr('linear', e, self.BASE_LR, self.SWA_LR,
                                      self.ANNEAL_EPOCHS) for e in range(1, 13)]
        torch.testing.assert_close(
            torch.tensor(got, dtype=torch.float64),
            torch.tensor(expected, dtype=torch.float64))
        self.assertEqual(got[9], self.SWA_LR, "lr must hit swa_lr at e=anneal_epochs")
        self.assertEqual(got[11], self.SWA_LR)

    def test_cos_and_linear_branches_differ(self):
        """Mid-anneal the two strategies produce different lrs (real branches)."""
        _, cos_sched = self._make_scheduler('cos')
        _, lin_sched = self._make_scheduler('linear')
        for _ in range(3):
            cos_sched.step()
            lin_sched.step()
        cos_lr = cos_sched.get_last_lr()[0]
        lin_lr = lin_sched.get_last_lr()[0]
        # e=3: cos = 0.05*0.20610737385376343 + 0.1*0.79389... ~ 0.0896946
        #      linear = 0.05*0.3 + 0.1*0.7 = 0.085
        self.assertAlmostEqual(lin_lr, 0.085, places=12)
        self.assertGreater(abs(cos_lr - lin_lr), 1e-4,
                           "cos and linear must differ mid-anneal")
        # Each branch matches its own closed form at e=3.
        self.assertAlmostEqual(
            cos_lr,
            self._expected_lr('cos', 3, self.BASE_LR, self.SWA_LR, self.ANNEAL_EPOCHS),
            places=12)

    def test_swa_lr_reached_and_held(self):
        """swa_lr takes effect: reached at e=anneal_epochs, held afterwards."""
        opt, sched = self._make_scheduler('linear')
        for _ in range(self.ANNEAL_EPOCHS):
            sched.step()
        self.assertEqual(sched.get_last_lr(), [self.SWA_LR])
        for _ in range(3):
            sched.step()
            self.assertEqual(sched.get_last_lr(), [self.SWA_LR],
                             "lr must stay pinned at swa_lr after annealing")
        # The optimizer param group actually carries the pinned lr.
        self.assertEqual(opt.param_groups[0]['lr'], self.SWA_LR)

    def test_anneal_epochs_zero_immediate_swa_lr(self):
        """Boundary anneal_epochs=0: swa_lr is in effect from the start."""
        opt, sched = self._make_scheduler('cos', anneal_epochs=0)
        self.assertEqual(sched.get_last_lr(), [self.SWA_LR],
                         "with anneal_epochs=0 the initial lr is already swa_lr")
        opt.step()
        sched.step()
        self.assertEqual(sched.get_last_lr(), [self.SWA_LR])

    def test_swa_lr_written_to_param_groups(self):
        """Construction writes swa_lr into every param group (scalar form)."""
        opt, sched = self._make_scheduler('cos', swa_lr=0.05)
        self.assertEqual(opt.param_groups[0]['swa_lr'], 0.05)
        # and get_lr honours it: after full anneal the lr equals swa_lr
        for _ in range(self.ANNEAL_EPOCHS):
            sched.step()
        self.assertEqual(sched.get_last_lr(), [0.05])

    def test_multi_group_swa_lr_list(self):
        """A swa_lr list anneals each group independently to its own target."""
        swa_lrs = [0.05, 0.01]
        base_lrs = [0.1, 0.2]
        opt = torch.optim.SGD([
            {'params': [torch.nn.Parameter(torch.zeros(1), requires_grad=True)],
             'lr': base_lrs[0]},
            {'params': [torch.nn.Parameter(torch.zeros(1), requires_grad=True)],
             'lr': base_lrs[1]},
        ])
        sched = torch.optim.swa_utils.SWALR(
            opt, swa_lr=swa_lrs, anneal_epochs=self.ANNEAL_EPOCHS,
            anneal_strategy='linear')
        opt.step()
        self.assertEqual([g['swa_lr'] for g in opt.param_groups], swa_lrs)
        for e in range(1, 4):
            sched.step()
            expected = [self._expected_lr('linear', e, b, s, self.ANNEAL_EPOCHS)
                        for b, s in zip(base_lrs, swa_lrs)]
            torch.testing.assert_close(
                torch.tensor(sched.get_last_lr(), dtype=torch.float64),
                torch.tensor(expected, dtype=torch.float64))

    def test_direct_get_lr_warns_to_use_get_last_lr(self):
        """Direct get_lr() outside step() warns and returns the base lr at init."""
        opt = torch.optim.SGD(
            [torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.1)
        swa = torch.optim.swa_utils.SWALR(opt, swa_lr=0.05)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            lr = swa.get_lr()
        self.assertEqual(lr, [0.1],
                         "at init (t=0) get_lr returns the untouched base lr")
        self.assertTrue(any(issubclass(w.category, UserWarning) and
                            'get_last_lr' in str(w.message) for w in caught),
                        "calling get_lr outside step must warn to use get_last_lr")

    def test_invalid_anneal_strategy_raises_value_error(self):
        """Bad anneal_strategy/anneal_epochs fail construction with ValueError."""
        opt = torch.optim.SGD(
            [torch.nn.Parameter(torch.zeros(1), requires_grad=True)], lr=0.1)
        with self.assertRaises(ValueError):
            torch.optim.swa_utils.SWALR(opt, swa_lr=0.05, anneal_strategy='quadratic')
        opt2 = torch.optim.SGD(
            [torch.nn.Parameter(torch.zeros(1), requires_grad=True)], lr=0.1)
        with self.assertRaises(ValueError):
            torch.optim.swa_utils.SWALR(opt2, swa_lr=0.05, anneal_epochs=-1)


if __name__ == "__main__":
    run_tests()
