# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.optim.Optimizer.add_param_group 接口功能正确性
API 名称：torch.optim.Optimizer.add_param_group
API 签名：Optimizer.add_param_group(self, param_group: dict[str, Any]) -> None
          Add a param group to the optimizer's `param_groups` (used to make
          previously frozen layers trainable during fine-tuning). This is a
          pure-Python state mutator: it appends the given dict to
          `self.param_groups`, normalizing `params` to a list of Tensors and
          merging missing hyperparameters from `self.defaults`. It holds the
          caller's Tensor references (no copy) and performs pre-append
          validation (type / leaf / required-default / duplicate / name-mix).

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                     |
|------------------|--------------------------------------------------------------|----------------------------------------------|
| 空/非空          | 单 Tensor 入参 / 空 list / 非空 list                          | 已覆盖（Tensor→list、空 list 合法组、多参数组） |
| 枚举选项         | params 支持的容器形态：Tensor、list、set（非法）              | 已覆盖（Tensor/list 合法，set 抛 TypeError） |
| 参数类型         | param_group 必须为 dict；组内元素必须为 Tensor           | 已覆盖（非 dict / 非 Tensor 抛 TypeError）  |
| 传参与不传参     | 组内是否显式提供默认超参（缺省由 defaults 补齐）             | 已覆盖（默认补齐 / 显式覆盖）                 |
| 等价类/边界值    | 单参数组、多参数组追加；命名参数 all-or-none；跨组命名一致性  | 已覆盖                                         |
| 正常传参场景     | 追加组后 len(param_groups) 增 1、参数为同一引用、数据未变     | 已覆盖（引用同一性 assertIs + 数据 assert_close）|
| 异常传参场景     | 非 dict / set / 非 Tensor / 非叶张量 / 缺 required 默认 / 跨组重复 / 命名混用 | 已覆盖（各自抛既定异常） |
| 混合设备类型     | 纯 Python 参数组结构操作，不涉及张量设备迁移                 | 未覆盖（add_param_group 只整理结构与引用，无跨设备计算） |

未覆盖项及原因：
- 混合设备类型：add_param_group 仅维护 param_groups 列表结构与参数引用，不含设备上数值计算，
  无法构造 NPU/CPU 混合设备输入的数值对照，故不涉及设备迁移验证。
- required 缺省校验用自定义 default（lr 设为 required）触发，标准 SGD 的 defaults 无 required，故用
  手动构造 Optimizer 的 default 场景验证该分支。
- 符号 / 常量张量（fake tensor）等非 Parameter 场景：add_param_group 只接受真实 torch.Tensor，
  不引入 tensor subclass。

注意：本 API 无浮点数值计算（不产生 float 输出），故不适用 CPU Golden 数值对照；断言语义重
     「同一对象引用（assertIs）」与「数据未被拷贝（assert_close）」两条契约。
"""
import os
import unittest
import warnings

import torch
from torch.optim.optimizer import required

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
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


# A tiny concrete optimizer whose defaults include a required hyperparameter
# (lr). Its `step` only consumes stored hyperparams so it can be constructed
# cheaply for add_param_group introspection.
class _MiniOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, *, my_flag=False, differentiable=False):
        defaults = {"lr": lr, "my_flag": my_flag, "differentiable": differentiable}
        super().__init__(params, defaults)

    @torch._disable_dynamo
    def step(self, closure=None):
        if closure is not None:
            closure()
        return None


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestOptimizerAddParamGroup(TestCase):
    """Test cases for torch.optim.Optimizer.add_param_group (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_add_param_group_appends_and_returns_none(self):
        """Adding a group appends to param_groups and add_param_group returns None."""
        opt = _MiniOptimizer([torch.randn(2, 3)], lr=0.1)
        before = len(opt.param_groups)
        new_param = torch.randn(4, 5)
        result = opt.add_param_group({"params": new_param, "lr": 0.01})
        self.assertIsNone(result)
        self.assertEqual(len(opt.param_groups), before + 1)

    def test_add_param_group_holds_same_reference(self):
        """The optimizer holds the exact caller-provided Tensor object (no copy)."""
        p = torch.randn(3, 3)
        opt = _MiniOptimizer([torch.randn(2, 2)], lr=0.1)
        opt.add_param_group({"params": p})
        group = opt.param_groups[-1]
        self.assertIs(group["params"][0], p)

    def test_add_param_group_does_not_copy_data(self):
        """Registering a param must not mutate its data (incl. NaN)."""
        p = torch.tensor([1.0, float("nan"), 3.0], requires_grad=True)
        before = p.detach().clone()
        opt = _MiniOptimizer([torch.randn(2)], lr=0.1)
        opt.add_param_group({"params": p})
        torch.testing.assert_close(opt.param_groups[-1]["params"][0], before, equal_nan=True)

    def test_tensor_params_normalized_to_list(self):
        """A bare Tensor in params is normalized into a single-element list."""
        p = torch.randn(2, 2)
        opt = _MiniOptimizer([torch.randn(1)], lr=0.1)
        opt.add_param_group({"params": p})
        group = opt.param_groups[-1]
        self.assertIsInstance(group["params"], list)
        self.assertEqual(len(group["params"]), 1)
        self.assertIs(group["params"][0], p)

    def test_default_hyrperparams_merged(self):
        """Missing hyperparams are filled from defaults; explicit values override."""
        opt = _MiniOptimizer([torch.randn(1)], lr=0.1)
        opt.add_param_group({"params": torch.randn(1)})  # no lr -> from defaults
        group = opt.param_groups[-1]
        self.assertEqual(group["lr"], 0.1)
        self.assertFalse(group["my_flag"])
        # explicit override
        opt.add_param_group({"params": torch.randn(1), "lr": 0.5, "my_flag": True})
        group2 = opt.param_groups[-1]
        self.assertEqual(group2["lr"], 0.5)
        self.assertTrue(group2["my_flag"])

    def test_non_dict_raises_type_error(self):
        """param_group must be a dict."""
        opt = _MiniOptimizer([torch.randn(1)], lr=0.1)
        with self.assertRaises(TypeError):
            opt.add_param_group([("params", torch.randn(1))])

    def test_set_params_raises_type_error(self):
        """Set params are rejected (unordered)."""
        p1, p2 = torch.randn(1), torch.randn(1)
        opt = _MiniOptimizer([torch.randn(1)], lr=0.1)
        with self.assertRaises(TypeError):
            opt.add_param_group({"params": {p1, p2}})

    def test_non_tensor_param_raises_type_error(self):
        """A non-Tensor element in params is rejected."""
        opt = _MiniOptimizer([torch.randn(1)], lr=0.1)
        with self.assertRaises(TypeError):
            opt.add_param_group({"params": [torch.randn(1), 42]})

    def test_non_leaf_param_raises_value_error(self):
        """A non-leaf Tensor cannot be optimized (unless differentiable)."""
        opt = _MiniOptimizer([torch.randn(1)], lr=0.1, my_flag=True)
        non_leaf = torch.randn(2, requires_grad=True).relu()
        with self.assertRaises(ValueError):
            opt.add_param_group({"params": non_leaf})
        # The leaf restriction is checked against self.defaults["differentiable"],
        # not the per-group value.
        opt_diff = _MiniOptimizer([torch.randn(1)], lr=0.1, differentiable=True)
        opt_diff.add_param_group({"params": non_leaf, "my_flag": True})
        self.assertEqual(len(opt_diff.param_groups), 2)

    def test_missing_required_default_raises_value_error(self):
        """A group missing a `required` default hyperparameter raises ValueError."""
        opt = _MiniOptimizer([torch.randn(1)], lr=0.1)
        # Simulate a hyperparameter whose value is mandatory (e.g. Adam's capturable
        # paths mark lr Required); the initial group already supplied lr, so a
        # subsequent group omitting it must fail.
        opt.defaults["lr"] = required
        with self.assertRaisesRegex(ValueError, "required optimization parameter"):
            opt.add_param_group({"params": torch.randn(1), "my_flag": True})

    def test_group_with_required_satisfied(self):
        """When the required hyperparameter is supplied, the group is accepted."""
        opt = _MiniOptimizer([torch.randn(1)], lr=0.1)
        opt.defaults["lr"] = required
        opt.add_param_group({"params": torch.randn(1), "lr": 0.001, "my_flag": True})
        self.assertEqual(len(opt.param_groups), 2)
        self.assertEqual(opt.param_groups[-1]["lr"], 0.001)

    def test_duplicate_param_within_group_warns(self):
        """Duplicate params in the same group raise a UserWarning."""
        p = torch.randn(1)
        opt = _MiniOptimizer([torch.randn(1)], lr=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            opt.add_param_group({"params": [p, p]})
        self.assertTrue(any(issubclass(x.category, UserWarning) for x in w))

    def test_duplicate_param_across_groups_raises_value_error(self):
        """A param already in another group cannot be added again."""
        p = torch.randn(1)
        opt = _MiniOptimizer([p], lr=0.1)
        with self.assertRaisesRegex(ValueError, "more than one parameter group"):
            opt.add_param_group({"params": p})

    def test_partial_named_params_raises_value_error(self):
        """All-or-none naming: a group mixing named and unnamed params is rejected."""
        named = ("w", torch.randn(2))
        opt = _MiniOptimizer([torch.randn(1)], lr=0.1)
        with self.assertRaisesRegex(ValueError, "Some param names are missing"):
            opt.add_param_group({"params": [named, torch.randn(2)]})

    def test_named_params_all_match(self):
        """Fully named group gets param_names recorded."""
        # The initial param group must already be named, otherwise adding a named
        # group fails the cross-group naming consistency check.
        opt = _MiniOptimizer([("init", torch.randn(1))], lr=0.1)
        opt.add_param_group({"params": [("a", torch.randn(2)), ("b", torch.randn(2))]})
        group = opt.param_groups[-1]
        self.assertEqual(group["param_names"], ["a", "b"])

    def test_named_mismatch_across_groups_raises_value_error(self):
        """A named group cannot be added when existing groups are unnamed."""
        opt = _MiniOptimizer([torch.randn(1)], lr=0.1)
        with self.assertRaisesRegex(ValueError, "with/without names"):
            opt.add_param_group({"params": [("a", torch.randn(2))]})


if __name__ == "__main__":
    run_tests()
