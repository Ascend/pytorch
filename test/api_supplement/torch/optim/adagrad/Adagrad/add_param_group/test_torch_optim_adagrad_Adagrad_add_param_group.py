# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.adagrad.Adagrad.add_param_group 接口功能正确性
API 名称：torch.optim.adagrad.Adagrad.add_param_group
API 签名：add_param_group(self, param_group: dict) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 新增含参数张量的参数组 | 已覆盖（test_add_param_group_basic） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | dict 必含 'params'；params 可为 list/单 Tensor | 已覆盖（_tensor_wrapped/_non_dict/_non_tensor/_set_params） |
| 传参与不传参 | 组内超参省略（走 Adagrad defaults）与显式传参 | 已覆盖（_defaults_merged_from_adagrad/_explicit_hyperparams_kept） |
| 等价类/边界值 | 重复参数、非叶子张量、新组 state 惰性为空 | 已覆盖（_duplicate/_non_leaf/_new_group_state_empty） |
| 正常传参场景 | 追加后长度 +1、引用同一、数据未拷贝（NaN 探针）、新组独立 step | 已覆盖（_basic/_step_updates_new_group_independently） |
| 异常传参场景 | 非 dict/缺 params/set/非 Tensor/非叶/重复 | 已覆盖（六个独立用例） |
| 混合设备类型 | 参数在当前设备（NPU/CUDA）、step Golden 对照 | 已覆盖（_add_group_on_current_device/_step_updates_new_group_independently） |

未覆盖项及原因：
- 命名参数（('name', param) 形态）：当前环境 Adagrad 基类路径未启用命名参数契约，属实验特性，不测。
- 本接口为纯状态管理，无数值执行语义；step 相关数值语义由 CPU Golden 对照验证。
- 实测说明：Adagrad 在构造函数中即时初始化全部参数的 state（step/sum），但 add_param_group
  追加的新参数 state 为空且 step() 不会惰性初始化（直接 step 抛 KeyError 'sum'）；
  本文件用 _new_group_state_empty / _step_without_state_init_raises_keyerror 记录该行为，
  多组 step 用例按构造函数同款方式手动初始化新组 state 后验证独立更新。
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
class TestAdagradadd_param_group(TestCase):
    """Comprehensive cases for torch.optim.adagrad.Adagrad.add_param_group (NPU/CUDA portable).

    Adagrad inherits add_param_group from Optimizer, but this file still covers
    the full contract independently: hyperparameter merge against Adagrad's own
    defaults (lr_decay/eps=1e-10/initial_accumulator_value), reference identity
    (no copy), the full validation family, the Adagrad-specific eager/lazy state
    asymmetry and multi-group step behavior.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=0.01):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.Adagrad([p], lr=lr)

    def _init_state_like_ctor(self, opt, param):
        """Initialize state for a param added via add_param_group.

        Adagrad.__init__ eagerly initializes state (step/sum) for every param,
        but params appended later via add_param_group get an EMPTY state and
        Adagrad.step() does NOT lazily initialize it (verified: KeyError 'sum').
        This helper mirrors the ctor initialization so multi-group stepping can
        be exercised, exactly like the ctor would have done for group 1.
        """
        group = opt.param_groups[-1]
        opt.state[param]["step"] = torch.tensor(0.0)
        opt.state[param]["sum"] = torch.full_like(
            param, float(group["initial_accumulator_value"]))

    # ------------------------------------------------------------------
    # Normal paths
    # ------------------------------------------------------------------

    def test_add_param_group_basic(self):
        """A new group is appended; the optimizer holds the exact caller Tensor."""
        opt = self._make_optimizer()
        n0 = len(opt.param_groups)
        # NaN probe element: equal_nan=True below makes the copy check honest
        # even for NaN payloads.
        p2 = torch.nn.Parameter(
            torch.tensor([float("nan"), 1.0, 2.0], device=self.device),
            requires_grad=True)
        p2_data = p2.detach().clone()
        opt.add_param_group({"params": [p2], "lr": 0.05})
        self.assertEqual(len(opt.param_groups), n0 + 1)
        group = opt.param_groups[-1]
        # Reference identity: the optimizer must hold the exact Tensor object.
        self.assertIs(group["params"][0], p2, "must hold the exact caller Tensor (no copy)")
        # Data must not be mutated or copied by registration (NaN probe included).
        torch.testing.assert_close(group["params"][0].data, p2_data, equal_nan=True,
                                   msg="adding a group must not mutate/copy param data")

    def test_defaults_merged_from_adagrad(self):
        """A group omitting hyperparameters inherits Adagrad's own defaults."""
        opt = self._make_optimizer(lr=0.02)
        p2 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt.add_param_group({"params": [p2]})
        group = opt.param_groups[-1]
        # Adagrad-specific defaults (NOT SGD/Adam values) must be merged in.
        self.assertEqual(group["lr"], 0.02)  # lr propagates from the optimizer default
        self.assertEqual(group["lr_decay"], 0)
        self.assertEqual(group["eps"], 1e-10)
        self.assertEqual(group["weight_decay"], 0)
        self.assertEqual(group["initial_accumulator_value"], 0)
        self.assertIsNone(group["foreach"])
        self.assertFalse(group["maximize"])
        self.assertFalse(group["differentiable"])
        self.assertIsNone(group["fused"])

    def test_explicit_hyperparams_kept(self):
        """Explicit values in the group override the optimizer defaults."""
        opt = self._make_optimizer(lr=0.02)
        p2 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt.add_param_group({"params": [p2], "lr": 0.05, "lr_decay": 0.1,
                             "eps": 1e-8, "weight_decay": 0.2,
                             "initial_accumulator_value": 0.5})
        group = opt.param_groups[-1]
        self.assertEqual(group["lr"], 0.05)
        self.assertEqual(group["lr_decay"], 0.1)
        self.assertEqual(group["eps"], 1e-8)
        self.assertEqual(group["weight_decay"], 0.2)
        self.assertEqual(group["initial_accumulator_value"], 0.5)

    def test_tensor_params_wrapped(self):
        """A single Tensor for params is wrapped into a list automatically."""
        opt = self._make_optimizer()
        p2 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt.add_param_group({"params": p2})
        self.assertEqual(len(opt.param_groups), 2)
        self.assertIsInstance(opt.param_groups[-1]["params"], list)
        self.assertIs(opt.param_groups[-1]["params"][0], p2)

    def test_new_group_state_empty(self):
        """Adagrad-specific: params added later get an EMPTY state (lazy).

        The ctor eagerly initializes state for initial params; add_param_group
        does not. This asymmetry is observable: opt.state[p2] is an empty dict
        right after registration.
        """
        opt = self._make_optimizer()
        p1 = opt.param_groups[0]["params"][0]
        # Ctor-initialized param already has its state slots.
        self.assertIn("step", opt.state[p1])
        self.assertIn("sum", opt.state[p1])
        p2 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt.add_param_group({"params": [p2]})
        self.assertEqual(len(opt.state[p2]), 0,
                         "params appended via add_param_group start with empty state")

    def test_step_without_state_init_raises_keyerror(self):
        """Current-version contract: stepping a lazily-added group raises KeyError.

        Adagrad._init_group reads state['sum'] without lazy initialization,
        so a param added via add_param_group whose state was never initialized
        makes step() fail with KeyError('sum'). This documents the asymmetry
        against the eager ctor initialization (verified on torch 2.11).
        """
        opt = self._make_optimizer(lr=1.0)
        p2 = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt.add_param_group({"params": [p2], "lr": 1.0})
        p1 = opt.param_groups[0]["params"][0]
        p1.grad = torch.full((3,), 2.0, device=self.device)
        p2.grad = torch.full((3,), 4.0, device=self.device)
        with self.assertRaises(KeyError):
            opt.step()

    def test_step_updates_new_group_independently(self):
        """After adding a group, step updates both groups independently.

        State for the appended group is initialized exactly like the ctor
        would (see _init_state_like_ctor). Per-group lr must be honored:
        group-1 lr=1.0 and group-2 lr=0.5 with identical grads produce
        different parameter deltas. CPU result is the Golden reference for
        the device path.
        """

        def run(device):
            p1 = torch.nn.Parameter(torch.ones(3, device=device), requires_grad=True)
            p2 = torch.nn.Parameter(torch.ones(3, device=device), requires_grad=True)
            opt = torch.optim.Adagrad([p1], lr=1.0)
            opt.add_param_group({"params": [p2], "lr": 0.5})
            self._init_state_like_ctor(opt, p2)
            p1.grad = torch.full((3,), 2.0, device=device)
            p2.grad = torch.full((3,), 2.0, device=device)
            before1, before2 = p1.detach().clone(), p2.detach().clone()
            opt.step()
            return opt, p1, p2, before1, before2

        opt, p1, p2, before1, before2 = run(self.device)
        self.assertFalse(torch.equal(p1.detach(), before1), "group-1 param must be updated")
        self.assertFalse(torch.equal(p2.detach(), before2), "new group param must be updated")
        # The new group participates with its own state slot.
        self.assertIn(p2, opt.state)
        self.assertEqual(opt.state[p2]["step"].item(), 1.0)
        # Both groups saw the same grad, so the accumulator matches...
        torch.testing.assert_close(opt.state[p1]["sum"], opt.state[p2]["sum"])
        # ...but the per-group lr produces different parameter deltas.
        torch.testing.assert_close(p1.detach(), torch.zeros_like(p1))
        torch.testing.assert_close(p2.detach(), torch.full_like(p2, 0.5))

        # CPU Golden: identical construction on CPU must match the device path.
        _, p1c, p2c, _, _ = run("cpu")
        torch.testing.assert_close(p1.detach().cpu(), p1c.detach())
        torch.testing.assert_close(p2.detach().cpu(), p2c.detach())

    # ------------------------------------------------------------------
    # Validation family (each case independent)
    # ------------------------------------------------------------------

    def test_duplicate_param_raises(self):
        """A param already in the optimizer raises ValueError."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        with self.assertRaisesRegex(ValueError, "some parameters appear in more than one"):
            opt.add_param_group({"params": [p]})

    def test_missing_params_raises(self):
        """A group without the 'params' key raises KeyError."""
        opt = self._make_optimizer()
        with self.assertRaises(KeyError):
            opt.add_param_group({"lr": 0.01})

    def test_non_dict_raises(self):
        """A non-dict param_group raises TypeError."""
        opt = self._make_optimizer()
        with self.assertRaises(TypeError):
            opt.add_param_group(list(opt.param_groups[0]["params"]))

    def test_set_params_raises(self):
        """A set for params raises TypeError (stable ordering required)."""
        opt = self._make_optimizer()
        p2 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        with self.assertRaises(TypeError):
            opt.add_param_group({"params": {p2}})

    def test_non_tensor_param_raises(self):
        """A non-Tensor entry in params raises TypeError."""
        opt = self._make_optimizer()
        with self.assertRaisesRegex(TypeError, "optimizer can only optimize Tensors"):
            opt.add_param_group({"params": [1, 2]})

    def test_non_leaf_param_raises(self):
        """A non-leaf Tensor param raises ValueError."""
        opt = self._make_optimizer()
        leaf = torch.randn(3, device=self.device, requires_grad=True)
        non_leaf = leaf * 2
        with self.assertRaisesRegex(ValueError, "can't optimize a non-leaf Tensor"):
            opt.add_param_group({"params": [non_leaf]})

    # ------------------------------------------------------------------
    # Device portability check
    # ------------------------------------------------------------------

    def test_add_group_on_current_device(self):
        """Groups hold params created on the active (NPU/CUDA) device."""
        if self.device_name == 'cpu':
            self.skipTest("device-path test requires an accelerator")
        opt = self._make_optimizer()
        p2 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt.add_param_group({"params": [p2]})
        # Compare device types: torch.device('npu') str is 'npu' but a Tensor
        # reports 'npu:0' (indexed), so compare type, not the full string.
        self.assertEqual(opt.param_groups[-1]["params"][0].device.type, self.device.type)


if __name__ == "__main__":
    run_tests()
