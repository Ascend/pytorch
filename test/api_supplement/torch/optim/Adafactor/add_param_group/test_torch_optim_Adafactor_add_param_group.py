# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.Adafactor.add_param_group 接口功能正确性
API 名称：torch.optim.Adafactor.add_param_group
API 签名：add_param_group(self, param_group: dict) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 新增含参数张量的参数组 | 已覆盖（test_add_param_group_basic） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | dict 必含 'params'；params 可为 list/单 Tensor | 已覆盖（_tensor_wrapped/_non_dict/_non_tensor/_set_params） |
| 传参与不传参 | 组内超参省略（走 Adafactor defaults）与显式传参 | 已覆盖（_defaults_merged_from_adafactor/_explicit_hyperparams_kept） |
| 等价类/边界值 | 重复参数、非叶子张量、1-D/2-D 参数分桶 state | 已覆盖（_duplicate/_non_leaf/_step_new_group_state_layout） |
| 正常传参场景 | 追加后长度 +1、引用同一、数据未拷贝（NaN 探针）、新组独立 step | 已覆盖（_basic/_step_updates_new_group_independently） |
| 异常传参场景 | 非 dict/缺 params/set/非 Tensor/非叶/重复 | 已覆盖（六个独立用例） |
| 混合设备类型 | 参数张量位于当前设备；step 数值 CPU Golden 对照 | 已覆盖（_add_group_on_current_device/_step_updates_new_group_independently） |

未覆盖项及原因：
- 命名参数（('name', param) 形态）：当前环境 Adafactor 基类路径未启用命名参数契约，属实验特性，不测。
- 本接口为纯状态管理，无数值执行语义；step 数值语义由 CPU Golden 对照验证。
- 实测说明：Adafactor 与 Adagrad 不同，其 state 完全惰性（构造期为空），add_param_group
  追加的新参数在首次 step() 时自动按 1-D/2-D 维度分桶初始化 variance/row_var/col_var，
  无需手动初始化，多组 step 用例直接 step 即可。
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
class TestAdafactoradd_param_group(TestCase):
    """Comprehensive cases for torch.optim.Adafactor.add_param_group (NPU/CUDA portable).

    Adafactor inherits add_param_group from Optimizer, but this file still
    covers the full contract independently: hyperparameter merge against
    Adafactor's own defaults (beta2_decay=-0.8, eps=(None, 0.001), d=1.0),
    reference identity (no copy), the full validation family, the Adafactor
    factored-state layout for 1-D vs 2-D params and multi-group step behavior.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=0.01):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.Adafactor([p], lr=lr)

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

    def test_defaults_merged_from_adafactor(self):
        """A group omitting hyperparameters inherits Adafactor's own defaults."""
        opt = self._make_optimizer(lr=0.02)
        p2 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt.add_param_group({"params": [p2]})
        group = opt.param_groups[-1]
        # Adafactor-specific defaults (NOT SGD/Adam values) must be merged in.
        self.assertEqual(group["lr"], 0.02)  # lr propagates from the optimizer default
        self.assertEqual(group["beta2_decay"], -0.8)
        self.assertEqual(group["eps"], (None, 0.001))  # tuple of (eps1, eps2)
        self.assertEqual(group["d"], 1.0)
        self.assertEqual(group["weight_decay"], 0.0)
        self.assertIsNone(group["foreach"])
        self.assertFalse(group["maximize"])

    def test_explicit_hyperparams_kept(self):
        """Explicit values in the group override the optimizer defaults."""
        opt = self._make_optimizer(lr=0.02)
        p2 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt.add_param_group({"params": [p2], "lr": 0.05, "beta2_decay": -0.5,
                             "eps": (1e-12, 1e-3), "d": 2.0,
                             "weight_decay": 0.1, "maximize": True})
        group = opt.param_groups[-1]
        self.assertEqual(group["lr"], 0.05)
        self.assertEqual(group["beta2_decay"], -0.5)
        self.assertEqual(group["eps"], (1e-12, 1e-3))
        self.assertEqual(group["d"], 2.0)
        self.assertEqual(group["weight_decay"], 0.1)
        self.assertTrue(group["maximize"])

    def test_tensor_params_wrapped(self):
        """A single Tensor for params is wrapped into a list automatically."""
        opt = self._make_optimizer()
        p2 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt.add_param_group({"params": p2})
        self.assertEqual(len(opt.param_groups), 2)
        self.assertIsInstance(opt.param_groups[-1]["params"], list)
        self.assertIs(opt.param_groups[-1]["params"][0], p2)

    def test_step_updates_new_group_independently(self):
        """After adding a group, step updates both groups with per-group lr.

        Unlike Adagrad, Adafactor lazily initializes state at the first step,
        so the appended group steps directly without manual state setup.
        Group-1 lr=0.01 and group-2 lr=0.02 with identical grads produce
        different parameter deltas (Adafactor update scales with lr through
        rho_t and alpha). CPU result is the Golden reference for the device path.
        """

        def run(device):
            p1 = torch.nn.Parameter(torch.ones(3, device=device), requires_grad=True)
            p2 = torch.nn.Parameter(torch.ones(3, device=device), requires_grad=True)
            opt = torch.optim.Adafactor([p1], lr=0.01)
            opt.add_param_group({"params": [p2], "lr": 0.02})
            p1.grad = torch.full((3,), 2.0, device=device)
            p2.grad = torch.full((3,), 2.0, device=device)
            before1, before2 = p1.detach().clone(), p2.detach().clone()
            opt.step()
            return opt, p1, p2, before1, before2

        opt, p1, p2, before1, before2 = run(self.device)
        self.assertFalse(torch.equal(p1.detach(), before1), "group-1 param must be updated")
        self.assertFalse(torch.equal(p2.detach(), before2), "new group param must be updated")
        # The new group participates with its own state slot and step counter.
        self.assertIn(p2, opt.state)
        self.assertEqual(opt.state[p2]["step"].item(), 1.0)
        self.assertIn("variance", opt.state[p2])
        # Per-group lr: different lr values must give different deltas for the
        # same grad, proving group hyperparameters are honored independently.
        # (alpha scales with min(lr, 1/sqrt(t)); note the variance buffers are
        # lr-independent and are expected to MATCH for identical grads.)
        self.assertFalse(torch.equal(p1.detach(), p2.detach()),
                         "distinct per-group lr must produce distinct deltas")
        torch.testing.assert_close(opt.state[p1]["variance"],
                                   opt.state[p2]["variance"])

        # CPU Golden: identical construction on CPU must match the device path.
        _, p1c, p2c, _, _ = run("cpu")
        torch.testing.assert_close(p1.detach().cpu(), p1c.detach())
        torch.testing.assert_close(p2.detach().cpu(), p2c.detach())

    def test_step_new_group_state_layout(self):
        """Factored state layout: 1-D params get 'variance', 2-D get row/col vars."""
        p1 = torch.nn.Parameter(torch.randn(4, device=self.device), requires_grad=True)
        opt = torch.optim.Adafactor([p1], lr=0.01)
        vec = torch.nn.Parameter(torch.randn(5, device=self.device), requires_grad=True)
        mat = torch.nn.Parameter(torch.randn(2, 5, device=self.device), requires_grad=True)
        opt.add_param_group({"params": [vec, mat]})
        p1.grad = torch.randn_like(p1)
        vec.grad = torch.randn_like(vec)
        mat.grad = torch.randn_like(mat)
        opt.step()
        # 1-D param: single 'variance' buffer of the same shape.
        self.assertIn("variance", opt.state[vec])
        self.assertEqual(opt.state[vec]["variance"].shape, vec.shape)
        self.assertNotIn("row_var", opt.state[vec])
        # 2-D param: factored second moment (row_var/col_var), no full variance.
        self.assertIn("row_var", opt.state[mat])
        self.assertIn("col_var", opt.state[mat])
        self.assertNotIn("variance", opt.state[mat])
        self.assertEqual(opt.state[mat]["row_var"].shape, (2, 1))
        self.assertEqual(opt.state[mat]["col_var"].shape, (1, 5))

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
