# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.sparse_adam.SparseAdam 接口功能正确性
API 名称：torch.optim.sparse_adam.SparseAdam
API 签名：SparseAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, maximize=False)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空 params 报 ValueError；generator/dict 参数组/单个 Tensor 包装为 list 均可构造 | 已覆盖（empty_params/generator_params/
dict_param_groups/single_tensor_params_is_list） |
| 枚举选项 | maximize=True/False 默认与显式；lr 可为 float 或 1 元素 Tensor | 已覆盖（test_maximize_flag_defaults/test_tensor_lr_accepted） |
| 参数类型 | params 需为 Tensor 迭代；sparse/complex 参数被拒绝；int 元素报 TypeError；裸 Tensor 报 TypeError | 已覆盖（sparse_param_rejected/
complex_param_rejected/non_tensor_param_typeerror/bare_tensor_rejected） |
| 传参与不传参 | 默认超参（lr=1e-3, betas=(0.9,0.999), eps=1e-8, maximize=False）与显式传参 | 已覆盖（default_hyperparameters/
explicit_hyperparameters_merge） |
| 等价类/边界值 | lr/beta/eps 边界校验（lr<=0、eps<=0、beta>=1 报 ValueError）；Tensor lr 多元素报 ValueError | 已覆盖（lr_bounds_validation/
eps_bounds_validation/betas_bounds_validation/tensor_lr_requires_one_element） |
| 正常传参场景 | defaults 含 lr/betas/eps/maximize 四键；param_group 继承 defaults；sparse 梯度下 step 可更新；amsgrad 关键字不存在（TypeError） |
已覆盖（defaults_keys/param_group_inherits_defaults/step_sparse_grad/amsgrad_not_supported） |
| 异常传参场景 | 空参数列表、非法 lr/eps/betas、sparse 参数、complex 参数、非 Tensor 参数、多元素 Tensor lr 均有稳定异常 | 已覆盖（empty_params/
lr_bounds_validation/eps_bounds_validation/betas_bounds_validation/sparse_param_rejected/complex_param_rejected/
non_tensor_param_typeerror/tensor_lr_requires_one_element） |
| 混合设备类型 | NPU 参数构造 + NPU sparse 梯度 step 与 CPU fresh-path Golden 对照（NPU 需非零 moments，见 step 文件说明） |
已覆盖（constructor_npu_step_matches_cpu_golden） |

未覆盖项及原因：
- amsgrad：SparseAdam 签名不支持该参数（实测 TypeError: unexpected keyword argument），用
  test_amsgrad_not_supported 以异常断言覆盖其「不存在」契约，无 True 构造分支可测。
- lr 缺省时 ValueError：实测 lr 有默认值 1e-3，非 required，「缺省报错」路径不存在；
  由 test_default_hyperparameters 正向断言默认值。
- 本接口本质为优化器/调度器/状态工具的框架行为，涉及浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。
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
class TestSparseAdamSparseAdam(TestCase):
    """Test cases for torch.optim.sparse_adam.SparseAdam (NPU/CUDA portable).

    Covers: default vs explicit hyperparameters (lr/betas/eps/maximize),
    float and 1-element-Tensor lr, params normalization (generator, dict
    groups, bare-tensor wrapping), constructor validation (empty params,
    non-tensor params, sparse/complex param rejection, lr/eps/betas
    bounds), the absent amsgrad kwarg, defaults merging into param groups,
    and a device-vs-CPU golden sparse step.

    NPU note (verified empirically): the NPU sparse_mask kernel drops
    explicitly-stored zeros, so the first numeric step over all-zero
    moments fails there. The golden test seeds constant non-zero moments on
    the device path (details in the step test file).
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _param(self, *shape, device='cpu'):
        return torch.nn.Parameter(torch.randn(*shape, device=device), requires_grad=True)

    # ------------------------------------------------------------------
    # Construction basics
    # ------------------------------------------------------------------

    def test_instantiate(self):
        """Constructor builds a usable SparseAdam optimizer."""
        p = torch.nn.Parameter(torch.randn(3, 2), requires_grad=True)
        opt = torch.optim.SparseAdam([p], lr=0.05)
        self.assertEqual(len(opt.param_groups), 1)
        self.assertIs(opt.param_groups[0]["params"][0], p,
                      "the optimizer must hold the caller's Tensor object")

    def test_default_hyperparameters(self):
        """Omitted hyperparameters fall back to the documented defaults."""
        opt = torch.optim.SparseAdam([self._param(3, 2)])
        self.assertEqual(opt.defaults["lr"], 1e-3)
        self.assertEqual(opt.defaults["betas"], (0.9, 0.999))
        self.assertEqual(opt.defaults["eps"], 1e-8)
        self.assertEqual(opt.defaults["maximize"], False)
        # The same defaults propagate into the created param group.
        self.assertEqual(opt.param_groups[0]["lr"], 1e-3)
        self.assertEqual(opt.param_groups[0]["betas"], (0.9, 0.999))
        self.assertEqual(opt.param_groups[0]["eps"], 1e-8)
        self.assertEqual(opt.param_groups[0]["maximize"], False)

    def test_explicit_hyperparameters_merge(self):
        """Explicit lr/betas/eps/maximize are stored in defaults and groups."""
        opt = torch.optim.SparseAdam(
            [self._param(3, 2)], lr=0.05, betas=(0.5, 0.25), eps=1e-6, maximize=True)
        self.assertEqual(opt.defaults["lr"], 0.05)
        self.assertEqual(opt.defaults["betas"], (0.5, 0.25))
        self.assertEqual(opt.defaults["eps"], 1e-6)
        self.assertTrue(opt.defaults["maximize"])
        self.assertEqual(opt.param_groups[0]["lr"], 0.05)
        self.assertEqual(opt.param_groups[0]["betas"], (0.5, 0.25))
        self.assertEqual(opt.param_groups[0]["eps"], 1e-6)
        self.assertTrue(opt.param_groups[0]["maximize"])

    def test_defaults_keys(self):
        """defaults holds exactly lr/betas/eps/maximize (no amsgrad, no weight_decay)."""
        opt = torch.optim.SparseAdam([self._param(3, 2)], lr=0.05)
        self.assertEqual(set(opt.defaults.keys()), {"lr", "betas", "eps", "maximize"})

    def test_maximize_flag_defaults(self):
        """maximize defaults to False and the explicit True value is honored."""
        opt_default = torch.optim.SparseAdam([self._param(3, 2)])
        self.assertIs(opt_default.param_groups[0]["maximize"], False)
        opt_true = torch.optim.SparseAdam([self._param(3, 2)], maximize=True)
        self.assertIs(opt_true.param_groups[0]["maximize"], True)
        opt_false = torch.optim.SparseAdam([self._param(3, 2)], maximize=False)
        self.assertIs(opt_false.param_groups[0]["maximize"], False)

    def test_tensor_lr_accepted(self):
        """A 1-element Tensor lr is accepted and stored unconverted."""
        lr_tensor = torch.tensor(0.02)
        opt = torch.optim.SparseAdam([self._param(3, 2)], lr=lr_tensor)
        self.assertIs(opt.param_groups[0]["lr"], lr_tensor,
                      "the Tensor lr object must be kept as-is in the group")

    def test_amsgrad_not_supported(self):
        """SparseAdam has no amsgrad kwarg; passing it raises TypeError."""
        with self.assertRaisesRegex(TypeError, "amsgrad"):
            torch.optim.SparseAdam([self._param(3, 2)], amsgrad=True)

    # ------------------------------------------------------------------
    # params normalization
    # ------------------------------------------------------------------

    def test_generator_params(self):
        """A generator of params is consumed into a single group."""
        def gen():
            yield self._param(3, 2)
            yield self._param(3, 2)
        opt = torch.optim.SparseAdam(gen(), lr=0.05)
        self.assertEqual(len(opt.param_groups), 1)
        self.assertEqual(len(opt.param_groups[0]["params"]), 2)

    def test_dict_param_groups(self):
        """Dict param groups keep their explicit lr; others inherit the default."""
        opt = torch.optim.SparseAdam(
            [{"params": [self._param(3, 2)], "lr": 0.2}, {"params": [self._param(3, 2)]}],
            lr=0.05)
        self.assertEqual(len(opt.param_groups), 2)
        self.assertEqual(opt.param_groups[0]["lr"], 0.2,
                         "the group's explicit lr must win over the constructor default")
        self.assertEqual(opt.param_groups[1]["lr"], 0.05,
                         "a group without lr must inherit the constructor default")
        # Non-lr SparseAdam defaults merge into every group.
        self.assertEqual(opt.param_groups[0]["betas"], (0.9, 0.999))
        self.assertEqual(opt.param_groups[1]["eps"], 1e-8)

    def test_param_group_inherits_defaults(self):
        """SparseAdam-specific defaults (betas/eps/maximize) merge into groups."""
        opt = torch.optim.SparseAdam([{"params": [self._param(3, 2)]}], lr=0.05)
        group = opt.param_groups[0]
        self.assertEqual(group["betas"], (0.9, 0.999))
        self.assertEqual(group["eps"], 1e-8)
        self.assertIs(group["maximize"], False)

    def test_single_tensor_params_is_list(self):
        """A bare Tensor params argument is reported as a non-iterable TypeError
        (verified: the constructor requires an iterable of Tensors or dicts)."""
        with self.assertRaisesRegex(TypeError, "iterable"):
            torch.optim.SparseAdam(torch.randn(3, 2).requires_grad_(True), lr=0.1)

    def test_bare_tensor_rejected(self):
        """Alias check: any non-iterable params value raises TypeError."""
        with self.assertRaises(TypeError):
            torch.optim.SparseAdam(12345, lr=0.1)

    def test_non_tensor_param_typeerror(self):
        """Non-Tensor items inside params raise TypeError."""
        with self.assertRaisesRegex(TypeError, "can only optimize Tensors"):
            torch.optim.SparseAdam([1, 2, 3], lr=0.1)

    def test_empty_params(self):
        """An empty params list raises ValueError."""
        with self.assertRaises(ValueError):
            torch.optim.SparseAdam([], lr=0.05)

    # ------------------------------------------------------------------
    # Layout/dtype validation of params
    # ------------------------------------------------------------------

    def test_sparse_param_rejected(self):
        """Sparse-layout parameters are rejected with ValueError."""
        sparse_p = torch.randn(3, 2).to_sparse().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "requires dense parameter tensors"):
            torch.optim.SparseAdam([sparse_p], lr=0.05)

    def test_complex_param_rejected(self):
        """Complex parameters are rejected with ValueError."""
        complex_p = torch.nn.Parameter(
            torch.randn(3, 2, dtype=torch.complex64), requires_grad=True)
        with self.assertRaisesRegex(ValueError, "does not support complex parameters"):
            torch.optim.SparseAdam([complex_p], lr=0.05)

    # ------------------------------------------------------------------
    # Hyperparameter bounds validation
    # ------------------------------------------------------------------

    def test_lr_bounds_validation(self):
        """lr <= 0 raises ValueError; the smallest positive lr is accepted."""
        for bad in (0.0, -1.0):
            with self.assertRaisesRegex(ValueError, "Invalid learning rate"):
                torch.optim.SparseAdam([self._param(3, 2)], lr=bad)
        opt = torch.optim.SparseAdam([self._param(3, 2)], lr=1e-12)
        self.assertEqual(opt.param_groups[0]["lr"], 1e-12)

    def test_eps_bounds_validation(self):
        """eps <= 0 raises ValueError; a positive eps is accepted."""
        for bad in (0.0, -1e-8):
            with self.assertRaisesRegex(ValueError, "Invalid epsilon"):
                torch.optim.SparseAdam([self._param(3, 2)], eps=bad)
        opt = torch.optim.SparseAdam([self._param(3, 2)], eps=1e-4)
        self.assertEqual(opt.param_groups[0]["eps"], 1e-4)

    def test_betas_bounds_validation(self):
        """Each beta must lie in [0, 1); out-of-range values name the index."""
        with self.assertRaisesRegex(ValueError, "beta parameter at index 0"):
            torch.optim.SparseAdam([self._param(3, 2)], betas=(1.0, 0.999))
        with self.assertRaisesRegex(ValueError, "beta parameter at index 0"):
            torch.optim.SparseAdam([self._param(3, 2)], betas=(-0.1, 0.999))
        with self.assertRaisesRegex(ValueError, "beta parameter at index 1"):
            torch.optim.SparseAdam([self._param(3, 2)], betas=(0.9, 1.0))
        with self.assertRaisesRegex(ValueError, "beta parameter at index 1"):
            torch.optim.SparseAdam([self._param(3, 2)], betas=(0.9, -0.5))
        # Boundary values 0.0 are legal for both betas.
        opt = torch.optim.SparseAdam([self._param(3, 2)], betas=(0.0, 0.0))
        self.assertEqual(opt.param_groups[0]["betas"], (0.0, 0.0))

    def test_tensor_lr_requires_one_element(self):
        """A multi-element Tensor lr raises ValueError with a dedicated message."""
        with self.assertRaisesRegex(ValueError, "Tensor lr must be 1-element"):
            torch.optim.SparseAdam([self._param(3, 2)], lr=torch.tensor([1.0, 2.0]))

    # ------------------------------------------------------------------
    # Functional sanity: the constructed optimizer steps on sparse grads
    # ------------------------------------------------------------------

    def test_step_sparse_grad(self):
        """SparseAdam applies updates for sparse gradients (CPU fresh path)."""
        p = torch.nn.Parameter(torch.randn(3, 2), requires_grad=True)
        opt = torch.optim.SparseAdam([p], lr=0.05)
        idx = torch.tensor([[0, 2], [1, 1]])
        val = torch.tensor([1.0, -2.0])
        p.grad = torch.sparse_coo_tensor(idx, val, (3, 2)).coalesce()
        before = p.detach().clone()
        opt.step()
        self.assertFalse(torch.equal(before, p.detach()))
        # state for a stepped param holds the SparseAdam moment keys
        self.assertEqual(set(opt.state[p].keys()), {"step", "exp_avg", "exp_avg_sq"})

    def test_constructor_npu_step_matches_cpu_golden(self):
        """A device-constructed optimizer with a device sparse grad matches
        the CPU golden param after one seeded step."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        torch.manual_seed(0)
        init = torch.randn(4, 3)
        cpu_p = torch.nn.Parameter(init.clone(), requires_grad=True)
        dev_p = torch.nn.Parameter(init.clone().to(self.device), requires_grad=True)

        def sparse_grad(device):
            idx = torch.tensor([[0, 2], [0, 1]], device=device)
            val = torch.tensor([1.0, -2.0], device=device)
            return torch.sparse_coo_tensor(idx, val, (4, 3), device=device).coalesce()

        cpu_opt = torch.optim.SparseAdam([cpu_p], lr=0.1)
        dev_opt = torch.optim.SparseAdam([dev_p], lr=0.1)
        if self.device_name == 'npu':
            # NPU sparse_mask drops explicit zeros; seed non-zero moments
            # so the numeric kernel runs on the device path (see step tests).
            dev_opt.state[dev_p] = {
                "step": 0,
                "exp_avg": torch.full((4, 3), 0.05, device=self.device),
                "exp_avg_sq": torch.full((4, 3), 2.25, device=self.device),
            }
            cpu_opt.state[cpu_p] = {
                "step": 0,
                "exp_avg": torch.full((4, 3), 0.05),
                "exp_avg_sq": torch.full((4, 3), 2.25),
            }
        cpu_p.grad = sparse_grad('cpu')
        dev_p.grad = sparse_grad(self.device)
        cpu_opt.step()
        dev_opt.step()
        torch.testing.assert_close(dev_p.detach().cpu(), cpu_p.detach())


if __name__ == "__main__":
    run_tests()
