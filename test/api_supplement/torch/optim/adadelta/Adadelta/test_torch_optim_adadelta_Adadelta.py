# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.adadelta.Adadelta 接口功能正确性
API 名称：torch.optim.adadelta.Adadelta
API 签名：Adadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0, foreach=None, *, capturable=False, maximize=False)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空参数列表 ValueError；非空参数列表可实例化 | 已覆盖（test_empty_params/test_empty_params_error_message/test_instantiate） |
| 枚举选项 | foreach/capturable/maximize/differentiable 等开关入 defaults | 已覆盖（defaults_exact_values/
explicit_hyperparams_override） |
| 参数类型 | params 可为 list/dict 列表/生成器，不可为非可迭代对象；lr 可为 float 或 1 元素 Tensor | 已覆盖（generator_params_accepted/
non_iterable_params_raises/tensor_lr） |
| 传参与不传参 | 省略全部超参走 Adadelta defaults（lr=1.0/rho=0.9/eps=1e-6）；显式传参覆盖 | 已覆盖（defaults_exact_values/
explicit_hyperparams_override） |
| 等价类/边界值 | params 必传（缺失 TypeError，lr 非必传默认 1.0）；lr=0/rho=0/rho=1/eps=0 合法边界；
负 lr/rho∈[0,1] 外/负 eps/负 weight_decay ValueError；多元素 Tensor lr ValueError |
已覆盖（missing_params_raises/invalid_lr_raises/invalid_rho_raises/invalid_eps_raises/
invalid_weight_decay_raises/tensor_lr） |
| 正常传参场景 | 构造后 step 更新参数；多参数组每组独立超参且缺省项继承 Adadelta defaults | 已覆盖（step_updates/
multiple_param_groups/constructor_on_current_device） |
| 异常传参场景 | 空 params、缺 params、非可迭代 params、非法超参各自独立成例 | 已覆盖（六个独立异常用例） |
| 混合设备类型 | 参数张量位于当前设备（NPU/CUDA 可移植）可构造并可 step | 已覆盖（test_constructor_on_current_device） |

未覆盖项及原因：
- capturable=True 在 NPU 上不在支持设备列表（cuda/xpu/hpu），构造后 step 会失败，不作为功能断言，仅在覆盖表说明。
- 本接口为构造器（框架行为），浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验（见 step 文件）。

实测说明（与官方文档差异/确认点）：
- lr 非必传：默认 1.0（与 SGD 的 0.01/Adam 的 0.001 不同），仅 params 必传，缺失时 TypeError: missing 1 required positional argument: 'params'。
- 负 lr 抛 ValueError: Invalid learning rate: -0.1（Adadelta 允许 lr=0.0 边界）。
- 构造器对单个裸 Tensor 参数不自动包装（区别于 add_param_group），抛 TypeError: params argument ... should be an iterable of Tensors or dicts。
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
class TestAdadeltaAdadelta(TestCase):
    """Comprehensive cases for the torch.optim.adadelta.Adadelta constructor (NPU/CUDA portable).

    Covers: Adadelta's own defaults (lr=1.0/rho=0.9/eps=1e-6, distinct from
    SGD/Adam), explicit hyperparameter overrides, the required-params contract
    (lr is NOT required), boundary values (lr=0, rho in [0,1], eps=0), the
    full validation family (negative lr/rho/eps/weight_decay, multi-element
    Tensor lr, non-iterable params), multi-group construction with
    per-group hyperparams merged against Adadelta defaults, generator params
    and construction on the active device.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------
    # Existing basic paths (kept)
    # ------------------------------------------------------------------

    def test_instantiate(self):
        """Constructor builds a usable Adadelta optimizer."""
        p = torch.nn.Parameter(torch.randn(3), requires_grad=True)
        opt = torch.optim.Adadelta([p], lr=1.0)
        self.assertEqual(len(opt.param_groups), 1)

    def test_step_updates(self):
        """A step moves parameters in the update direction."""
        p = torch.nn.Parameter(torch.randn(3), requires_grad=True)
        opt = torch.optim.Adadelta([p], lr=1.0)
        p.grad = torch.ones_like(p)
        before = p.detach().clone()
        opt.step()

        self.assertFalse(torch.equal(before, p.detach()))

    def test_empty_params(self):
        """No params raises ValueError."""
        with self.assertRaises(ValueError):
            torch.optim.Adadelta([], lr=1.0)

    # ------------------------------------------------------------------
    # Defaults contract (Adadelta-specific values)
    # ------------------------------------------------------------------

    def test_defaults_exact_values(self):
        """Omitting every hyperparameter yields Adadelta's own defaults.

        Adadelta defaults are lr=1.0, rho=0.9, eps=1e-6 (NOT SGD's lr=0.01
        or Adam's lr=0.001); verified empirically on this build.
        """
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([p])
        expected = {
            "lr": 1.0,
            "rho": 0.9,
            "eps": 1e-06,
            "weight_decay": 0,
            "maximize": False,
            "capturable": False,
            "foreach": None,
            "differentiable": False,
        }
        self.assertEqual(opt.defaults, expected)
        # The first (and only) param group mirrors the defaults.
        group = opt.param_groups[0]
        for key, value in expected.items():
            self.assertEqual(group[key], value,
                             "group must carry Adadelta default %s=%r" % (key, value))

    def test_explicit_hyperparams_override(self):
        """Explicit hyperparameters are stored verbatim in defaults and groups."""
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta(
            [p], lr=0.5, rho=0.7, eps=1e-4, weight_decay=0.01,
            foreach=False, maximize=True, differentiable=True)
        expected = {
            "lr": 0.5,
            "rho": 0.7,
            "eps": 1e-4,
            "weight_decay": 0.01,
            "maximize": True,
            "capturable": False,
            "foreach": False,
            "differentiable": True,
        }
        self.assertEqual(opt.defaults, expected)
        group = opt.param_groups[0]
        for key, value in expected.items():
            self.assertEqual(group[key], value,
                             "group must carry explicit %s=%r" % (key, value))

    # ------------------------------------------------------------------
    # Required-params and empty-list contracts
    # ------------------------------------------------------------------

    def test_missing_params_raises(self):
        """params is the only required argument; omitting it raises TypeError.

        Measured: lr has a default (1.0), so passing only lr leaves the
        constructor without the single required positional argument.
        """
        with self.assertRaisesRegex(TypeError, "params"):
            torch.optim.Adadelta(lr=1.0)
        # Control: params alone (all hyperparams defaulted) constructs fine.
        p = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([p])
        self.assertEqual(opt.param_groups[0]["lr"], 1.0,
                         "lr must default to 1.0 without being passed")

    def test_empty_params_error_message(self):
        """An empty param list raises ValueError with the documented message."""
        with self.assertRaisesRegex(ValueError, "optimizer got an empty parameter list"):
            torch.optim.Adadelta([])

    # ------------------------------------------------------------------
    # Hyperparameter validation family (each case independent)
    # ------------------------------------------------------------------

    def test_invalid_lr_raises(self):
        """Negative lr raises ValueError; the lr=0.0 boundary is accepted."""
        with self.assertRaisesRegex(ValueError, "Invalid learning rate"):
            torch.optim.Adadelta([torch.nn.Parameter(torch.ones(2))], lr=-0.1)
        # Boundary: lr=0.0 must construct (0.0 <= lr).
        opt = torch.optim.Adadelta(
            [torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)],
            lr=0.0)
        self.assertEqual(opt.param_groups[0]["lr"], 0.0)

    def test_invalid_rho_raises(self):
        """rho outside [0, 1] raises ValueError; both endpoints are accepted."""
        for bad_rho in (-0.1, 1.5):
            with self.assertRaisesRegex(ValueError, "Invalid rho value"):
                torch.optim.Adadelta([torch.nn.Parameter(torch.ones(2))], rho=bad_rho)
        for good_rho in (0.0, 1.0):
            opt = torch.optim.Adadelta(
                [torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)],
                rho=good_rho)
            self.assertEqual(opt.param_groups[0]["rho"], good_rho)

    def test_invalid_eps_raises(self):
        """Negative eps raises ValueError; eps=0 is accepted."""
        with self.assertRaisesRegex(ValueError, "Invalid epsilon"):
            torch.optim.Adadelta([torch.nn.Parameter(torch.ones(2))], eps=-1e-6)
        opt = torch.optim.Adadelta(
            [torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)],
            eps=0.0)
        self.assertEqual(opt.param_groups[0]["eps"], 0.0)

    def test_invalid_weight_decay_raises(self):
        """Negative weight_decay raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Invalid weight_decay"):
            torch.optim.Adadelta([torch.nn.Parameter(torch.ones(2))], weight_decay=-0.1)

    def test_tensor_lr(self):
        """A 1-element Tensor lr is accepted (stored as-is); multi-element is rejected."""
        lr_t = torch.tensor(0.5)
        opt = torch.optim.Adadelta(
            [torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)],
            lr=lr_t)
        self.assertTrue(torch.is_tensor(opt.param_groups[0]["lr"]),
                        "a Tensor lr must be preserved as a Tensor")
        self.assertEqual(float(opt.param_groups[0]["lr"]), 0.5)
        with self.assertRaisesRegex(ValueError, "Tensor lr must be 1-element"):
            torch.optim.Adadelta([torch.nn.Parameter(torch.ones(2))],
                                 lr=torch.tensor([0.5, 0.6]))

    # ------------------------------------------------------------------
    # params typing
    # ------------------------------------------------------------------

    def test_non_iterable_params_raises(self):
        """A bare (non-iterable) params argument raises TypeError.

        Measured: unlike add_param_group, the constructor does NOT wrap a
        single Tensor into a list; it rejects it outright.
        """
        with self.assertRaisesRegex(TypeError, "should be an iterable of Tensors or dicts"):
            torch.optim.Adadelta(torch.nn.Parameter(torch.ones(2, device=self.device)))
        # Non-iterable plain object is rejected as well.
        with self.assertRaises(TypeError):
            torch.optim.Adadelta(123)

    def test_generator_params_accepted(self):
        """A generator of Tensors is consumed into a single param group."""
        def gen():
            yield torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
            yield torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)

        opt = torch.optim.Adadelta(gen())
        self.assertEqual(len(opt.param_groups), 1)
        self.assertEqual(len(opt.param_groups[0]["params"]), 2)

    # ------------------------------------------------------------------
    # Multiple param groups
    # ------------------------------------------------------------------

    def test_multiple_param_groups(self):
        """Each group keeps its explicit hyperparams; omitted keys inherit Adadelta defaults."""
        p1 = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        p2 = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([
            {"params": [p1], "lr": 0.5, "rho": 0.7},
            {"params": [p2], "eps": 1e-4},
        ])
        g1, g2 = opt.param_groups
        self.assertIs(g1["params"][0], p1, "group 1 must hold the exact caller Tensor")
        self.assertIs(g2["params"][0], p2, "group 2 must hold the exact caller Tensor")
        # Group 1: explicit values win.
        self.assertEqual(g1["lr"], 0.5)
        self.assertEqual(g1["rho"], 0.7)
        # Group 2: explicit eps wins, the rest inherit Adadelta's defaults.
        self.assertEqual(g2["eps"], 1e-4)
        self.assertEqual(g2["lr"], 1.0, "omitted lr must inherit Adadelta default 1.0")
        self.assertEqual(g2["rho"], 0.9, "omitted rho must inherit Adadelta default 0.9")
        self.assertEqual(g2["weight_decay"], 0)

    # ------------------------------------------------------------------
    # Device portability
    # ------------------------------------------------------------------

    def test_constructor_on_current_device(self):
        """Params created on the active (NPU/CUDA) device construct and step fine."""
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([p])
        self.assertEqual(opt.param_groups[0]["params"][0].device.type, self.device.type)
        p.grad = torch.ones_like(p)
        before = p.detach().clone()
        opt.step()
        self.assertFalse(torch.equal(before, p.detach()),
                         "device params must be updatable via step")


if __name__ == "__main__":
    run_tests()
