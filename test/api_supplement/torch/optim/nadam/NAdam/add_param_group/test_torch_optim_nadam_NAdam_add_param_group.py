# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.nadam.NAdam.add_param_group 接口功能正确性
API 名称：torch.optim.nadam.NAdam.add_param_group
API 签名：add_param_group(self, param_group: dict) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 新增含参数张量的参数组 | 已覆盖（test_add_param_group_basic） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | dict 必含 'params'；params 可为 list/单 Tensor | 已覆盖（_tensor_wrapped/_non_dict/_non_tensor/_set_params） |
| 传参与不传参 | 组内超参省略（走 NAdam defaults）与显式传参 | 已覆盖（_defaults_merged_from_nadam/_explicit_hyperparams_kept） |
| 等价类/边界值 | 重复参数、跨组重复、非叶子张量 | 已覆盖（_duplicate/_non_leaf） |
| 正常传参场景 | 追加后长度 +1、引用同一、数据未拷贝、新组可 step | 已覆盖（_basic/_step_updates_new_group） |
| 异常传参场景 | 非 dict/缺 params/set/非 Tensor/非叶/重复 | 已覆盖（六个独立用例） |
| 混合设备类型 | 参数张量位于当前设备（NPU/CUDA 可移植） | 已覆盖（_add_group_on_current_device） |

未覆盖项及原因：
- 命名参数（('name', param) 形态）：当前环境 NAdam 基类路径未启用命名参数契约，属实验特性，不测。
- 本接口为纯状态管理，无数值执行语义；step 相关数值语义在 NAdam/step 用例中覆盖。
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
class TestNAdamadd_param_group(TestCase):
    """Comprehensive cases for torch.optim.nadam.NAdam.add_param_group (NPU/CUDA portable).

    NAdam inherits add_param_group from Optimizer, but this file still covers
    the full contract independently: hyperparameter merge against NAdam's own
    defaults (betas=(0.9, 0.999)/eps=1e-8/momentum_decay=0.004/...), reference
    identity (no copy), the full validation family, single-Tensor wrapping and
    multi-group step behavior.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=0.01):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.NAdam([p], lr=lr)

    # ------------------------------------------------------------------
    # Normal paths
    # ------------------------------------------------------------------

    def test_add_param_group_basic(self):
        """A new group is appended; the optimizer holds the exact caller Tensor."""
        opt = self._make_optimizer()
        n0 = len(opt.param_groups)
        p2 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        p2_data = p2.detach().clone()
        opt.add_param_group({"params": [p2], "lr": 0.05})
        self.assertEqual(len(opt.param_groups), n0 + 1)
        group = opt.param_groups[-1]
        # Reference identity: the optimizer must hold the exact Tensor object.
        self.assertIs(group["params"][0], p2, "must hold the exact caller Tensor (no copy)")
        # Data must not be mutated or copied by registration (NaN probe included).
        torch.testing.assert_close(group["params"][0].data, p2_data, equal_nan=True,
                                   msg="adding a group must not mutate/copy param data")

    def test_add_param_group_nan_probe(self):
        """NaN/Inf content in the added param survives registration untouched."""
        opt = self._make_optimizer()
        p2 = torch.nn.Parameter(torch.tensor([float("nan"), float("inf"), 1.0],
                                             device=self.device))
        opt.add_param_group({"params": [p2]})
        kept = opt.param_groups[-1]["params"][0]
        self.assertIs(kept, p2)
        # equal_nan=True: NaN must stay NaN (not silently altered to 0).
        torch.testing.assert_close(kept.detach().cpu(),
                                   torch.tensor([float("nan"), float("inf"), 1.0]),
                                   equal_nan=True)

    def test_defaults_merged_from_nadam(self):
        """A group omitting hyperparameters inherits NAdam's own defaults."""
        opt = self._make_optimizer(lr=0.02)
        p2 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt.add_param_group({"params": [p2]})
        group = opt.param_groups[-1]
        # NAdam-specific defaults (NOT SGD/Adam values) must be merged in;
        # values verified against the installed build via python -c.
        self.assertEqual(group["betas"], (0.9, 0.999))
        self.assertEqual(group["eps"], 1e-08)
        self.assertEqual(group["momentum_decay"], 0.004)
        self.assertEqual(group["decoupled_weight_decay"], False)
        self.assertEqual(group["weight_decay"], 0)
        self.assertEqual(group["maximize"], False)
        self.assertEqual(group["lr"], 0.02)  # lr propagates from the optimizer default

    def test_explicit_hyperparams_kept(self):
        """Explicit values in the group override the optimizer defaults."""
        opt = self._make_optimizer(lr=0.02)
        p2 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt.add_param_group({"params": [p2], "lr": 0.05, "betas": (0.7, 0.99),
                             "eps": 1e-6, "momentum_decay": 0.01,
                             "weight_decay": 0.1, "decoupled_weight_decay": True})
        group = opt.param_groups[-1]
        self.assertEqual(group["lr"], 0.05)
        self.assertEqual(group["betas"], (0.7, 0.99))
        self.assertEqual(group["eps"], 1e-6)
        self.assertEqual(group["momentum_decay"], 0.01)
        self.assertEqual(group["weight_decay"], 0.1)
        self.assertEqual(group["decoupled_weight_decay"], True)

    def test_tensor_params_wrapped(self):
        """A single Tensor for params is wrapped into a list automatically."""
        opt = self._make_optimizer()
        p2 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt.add_param_group({"params": p2})
        self.assertEqual(len(opt.param_groups), 2)
        self.assertIsInstance(opt.param_groups[-1]["params"], list)
        self.assertIs(opt.param_groups[-1]["params"][0], p2)

    def test_step_updates_new_group(self):
        """After adding a group, step updates both groups independently."""
        p1 = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        p2 = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt = torch.optim.NAdam([p1], lr=0.1)
        opt.add_param_group({"params": [p2], "lr": 0.5})
        p1.grad = torch.full((3,), 2.0, device=self.device)
        p2.grad = torch.full((3,), 4.0, device=self.device)
        before1, before2 = p1.detach().clone(), p2.detach().clone()
        opt.step()
        self.assertFalse(torch.equal(p1.detach(), before1), "group-1 param must be updated")
        self.assertFalse(torch.equal(p2.detach(), before2), "new group param must be updated")
        # The new group participates with its own state slot.
        self.assertIn(p2, opt.state)
        # The new group's own lr drives a LARGER move than group 1 (0.5 vs 0.1
        # on the same all-positive gradient).
        d1 = (before1 - p1.detach()).abs().max().item()
        d2 = (before2 - p2.detach()).abs().max().item()
        self.assertGreater(d2, d1,
                           "lr=0.5 group must move further than the lr=0.1 group")

    # ------------------------------------------------------------------
    # Validation family (each case independent)
    # ------------------------------------------------------------------

    def test_add_param_group_duplicate(self):
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
