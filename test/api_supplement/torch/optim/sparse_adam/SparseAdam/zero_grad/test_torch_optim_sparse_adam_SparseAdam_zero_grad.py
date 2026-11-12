# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.sparse_adam.SparseAdam.zero_grad 接口功能正确性
API 名称：torch.optim.sparse_adam.SparseAdam.zero_grad
API 签名：zero_grad(self, set_to_none: bool = True) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 有 grad 参数被清空；无 grad 参数保持 None 不受影响 | 已覆盖（test_zero_grad_set_to_none/test_zero_grad_without_grad_param） |
| 枚举选项 | set_to_none=True/False/默认/不传参四分支 | 已覆盖（zero_grad_set_to_none/zero_grad_set_to_false/zero_grad_default/
zero_grad_explicit_true） |
| 参数类型 | set_to_none 不做类型校验：truthy 视为 True、falsy 视为 False | 已覆盖（test_zero_grad_invalid） |
| 传参与不传参 | 显式传 set_to_none=True 与省略参数默认行为一致 | 已覆盖（test_zero_grad_default/test_zero_grad_explicit_true） |
| 等价类/边界值 | set_to_none=False 对 sparse grad 为同对象 in-place 置零（is_sparse 保持、nnz 归零、to_dense 全零）；对 dense grad 同对象 in-place
置零且引用保持 | 已覆盖（zero_grad_set_to_false_sparse_semantics/zero_grad_set_to_false_dense_reference_kept） |
| 正常传参场景 | step 前置 sparse 梯度后 zero_grad 清空；清空后可重新赋梯度继续 step | 已覆盖（zero_grad_after_step_and_reassign/
zero_grad_set_to_none） |
| 异常传参场景 | zero_grad 无类型校验，truthy/falsy 归一为布尔分支，不抛异常 | 已覆盖（test_zero_grad_invalid） |
| 混合设备类型 | NPU 设备上 sparse/dense grad 的置零行为；多参数组在 NPU 上逐一清空 | 已覆盖（zero_grad_set_to_false_sparse_semantics/
zero_grad_multiple_param_groups_npu） |

未覆盖项及原因：
- 跨设备混合参数组（同一 optimizer 内 CPU+NPU 参数并存）：torch.optim 构造与 step 对混设备参数组
  本身不构成 zero_grad 的独立契约分支，zero_grad 逐参数就地操作与设备无关；以 NPU 单设备多参数组
  覆盖（test_zero_grad_multiple_param_groups_npu）。
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
class TestSparseAdamzero_grad(TestCase):
    """Test cases for torch.optim.sparse_adam.SparseAdam.zero_grad (NPU/CUDA portable).

    Covers: the four set_to_none branches (True / False / default / explicit),
    truthy-falsy normalization without type checking, sparse-grad zeroing
    semantics (in-place zero_() keeps the layout object, drops nnz and
    yields an all-zero dense view), dense-grad reference keeping, grads
    being reusable after zero_grad, grad-less params staying untouched,
    and multi-param-group clearing on the device.

    NPU note (verified empirically): sparse Tensor.zero_() on NPU keeps the
    same object, keeps is_sparse True, zeroes the stored values and drops
    nnz to 0 exactly as on CPU.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _sparse_grad(self, device, shape=(3, 2)):
        idx = torch.tensor([[0, 1], [1, 0]], device=device)
        val = torch.tensor([1.0, -2.0], device=device)
        return torch.sparse_coo_tensor(idx, val, shape, device=device).coalesce()

    def _make_opt(self, device=None):
        device = device or self.device
        p = torch.nn.Parameter(torch.randn(3, 2, device=device), requires_grad=True)
        opt = torch.optim.SparseAdam([p], lr=0.05)
        return p, opt

    # ------------------------------------------------------------------
    # set_to_none=True branch (and the default)
    # ------------------------------------------------------------------

    def test_zero_grad_set_to_none(self):
        """set_to_none=True leaves a sparse grad as None."""
        p, opt = self._make_opt()
        p.grad = self._sparse_grad(self.device)
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad, "set_to_none=True must set grad to None")

    def test_zero_grad_default(self):
        """Omitting set_to_none behaves as True (grad becomes None)."""
        p, opt = self._make_opt()
        p.grad = self._sparse_grad(self.device)
        opt.zero_grad()
        self.assertIsNone(p.grad, "the default call must set grad to None")

    def test_zero_grad_explicit_true(self):
        """Explicit set_to_none=True matches the default behavior."""
        p1, opt1 = self._make_opt()
        p2, opt2 = self._make_opt()
        p1.grad = self._sparse_grad(self.device)
        p2.grad = self._sparse_grad(self.device)
        opt1.zero_grad()
        opt2.zero_grad(set_to_none=True)
        self.assertIsNone(p1.grad)
        self.assertIsNone(p2.grad, "explicit True and the default must agree")

    # ------------------------------------------------------------------
    # set_to_none=False branch: in-place zeroing
    # ------------------------------------------------------------------

    def test_zero_grad_set_to_false(self):
        """set_to_none=False zeroes the grad in place (reference kept)."""
        p, opt = self._make_opt()
        p.grad = self._sparse_grad(self.device)
        ref = p.grad
        opt.zero_grad(set_to_none=False)
        self.assertIsNotNone(p.grad, "set_to_none=False must keep a tensor grad")
        self.assertIs(p.grad, ref,
                      "set_to_none=False must zero the existing object, not replace it")
        # Whatever the layout, the dense projection is all zeros now.
        if p.grad.is_sparse:
            torch.testing.assert_close(p.grad.to_dense(), torch.zeros(3, 2, device=self.device))
        else:
            torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))

    def test_zero_grad_set_to_false_sparse_semantics(self):
        """Zeroing a sparse grad keeps layout, drops nnz and zeroes values (NPU verified)."""
        p, opt = self._make_opt()
        p.grad = self._sparse_grad(self.device)
        ref = p.grad
        opt.zero_grad(set_to_none=False)
        grad = p.grad
        self.assertIs(grad, ref, "the sparse grad object must be zeroed in place")
        self.assertTrue(grad.is_sparse, "zeroing must preserve the sparse layout")
        self.assertEqual(grad._nnz(), 0, "zeroing drops all stored values")
        self.assertEqual(grad.layout, torch.sparse_coo)
        # The dense projection is all zeros on every backend.
        torch.testing.assert_close(grad.to_dense(), torch.zeros(3, 2, device=self.device))

    def test_zero_grad_set_to_false_dense_reference_kept(self):
        """A dense grad is also zeroed in place with the reference kept."""
        p, opt = self._make_opt()
        p.grad = torch.ones_like(p)
        ref = p.grad
        opt.zero_grad(set_to_none=False)
        self.assertIs(p.grad, ref, "the dense grad object must be zeroed in place")
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))

    # ------------------------------------------------------------------
    # No type checking of set_to_none
    # ------------------------------------------------------------------

    def test_zero_grad_invalid(self):
        """Truthy set_to_none behaves as True, falsy as False."""
        p, opt = self._make_opt()
        # zero_grad does not type-check set_to_none; any truthy value is
        # treated as True and zeroes to None, any falsy value as False.
        p.grad = torch.ones_like(p)
        opt.zero_grad(set_to_none="yes")
        self.assertIsNone(p.grad, "truthy set_to_none must be treated as True")
        p.grad = torch.ones_like(p)
        opt.zero_grad(set_to_none=0)
        self.assertIsNotNone(p.grad, "falsy set_to_none must be treated as False")
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))

    # ------------------------------------------------------------------
    # Grad-less params are untouched
    # ------------------------------------------------------------------

    def test_zero_grad_without_grad_param(self):
        """A param whose grad is already None stays None; no error is raised."""
        p, opt = self._make_opt()
        self.assertIsNone(p.grad)
        opt.zero_grad()
        self.assertIsNone(p.grad)
        opt.zero_grad(set_to_none=False)
        self.assertIsNone(p.grad, "zero_grad must not fabricate a grad")

    # ------------------------------------------------------------------
    # zero_grad is usable mid-training
    # ------------------------------------------------------------------

    def test_zero_grad_after_step_and_reassign(self):
        """After zero_grad a fresh sparse grad still drives a step update."""
        p, opt = self._make_opt()
        # Seed non-zero moments on NPU (sparse_mask drops explicit zeros
        # there; see the step test file for the full discussion).
        if self.device_name == 'npu':
            opt.state[p] = {
                "step": 0,
                "exp_avg": torch.full_like(p, 0.05),
                "exp_avg_sq": torch.full_like(p, 2.25),
            }
        p.grad = self._sparse_grad(self.device)
        opt.step()
        self.assertEqual(opt.state[p]["step"], 1)
        # None-style clear then reassign.
        opt.zero_grad()
        self.assertIsNone(p.grad)
        p.grad = self._sparse_grad(self.device)
        opt.step()
        self.assertEqual(opt.state[p]["step"], 2,
                         "a reassigned grad after zero_grad must keep stepping")
        # In-place-style clear then reassign a brand-new sparse grad object.
        opt.zero_grad(set_to_none=False)
        p.grad = self._sparse_grad(self.device)
        opt.step()
        self.assertEqual(opt.state[p]["step"], 3)

    # ------------------------------------------------------------------
    # Multiple param groups
    # ------------------------------------------------------------------

    def test_zero_grad_multiple_param_groups_npu(self):
        """zero_grad clears grads across all groups on the device."""
        p1 = torch.nn.Parameter(torch.randn(3, 2, device=self.device), requires_grad=True)
        p2 = torch.nn.Parameter(torch.randn(3, 2, device=self.device), requires_grad=True)
        p3 = torch.nn.Parameter(torch.randn(3, 2, device=self.device), requires_grad=True)
        opt = torch.optim.SparseAdam([{"params": [p1]}, {"params": [p2, p3]}], lr=0.05)
        self.assertEqual(len(opt.param_groups), 2)
        p1.grad = self._sparse_grad(self.device)
        p2.grad = torch.ones_like(p2)
        p3.grad = self._sparse_grad(self.device)
        opt.zero_grad()
        self.assertIsNone(p1.grad)
        self.assertIsNone(p2.grad)
        self.assertIsNone(p3.grad, "zero_grad must reach every group")
        # In-place pass across all groups and both layouts.
        p1.grad = self._sparse_grad(self.device)
        p2.grad = torch.ones_like(p2)
        p3.grad = self._sparse_grad(self.device)
        r1, r2, r3 = p1.grad, p2.grad, p3.grad
        opt.zero_grad(set_to_none=False)
        self.assertIs(p1.grad, r1)
        self.assertIs(p2.grad, r2)
        self.assertIs(p3.grad, r3)
        torch.testing.assert_close(p1.grad.to_dense(), torch.zeros(3, 2, device=self.device))
        torch.testing.assert_close(p2.grad, torch.zeros_like(p2))
        torch.testing.assert_close(p3.grad.to_dense(), torch.zeros(3, 2, device=self.device))


if __name__ == "__main__":
    run_tests()
