# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.Optimizer.zero_grad 接口功能正确性
API 名称：torch.optim.Optimizer.zero_grad
API 签名：zero_grad(self, set_to_none: bool = True) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 有 grad 参数被清空；grad 为 None 的参数不受影响且不报错 | 已覆盖（test_zero_grad_set_to_none/test_zero_grad_none_grad_unaffected） |
| 枚举选项 | set_to_none=True/False 全枚举；默认行为等价于显式 True | 已覆盖（zero_grad_set_to_none/zero_grad_set_to_false/
zero_grad_default_equals_explicit_true） |
| 参数类型 | set_to_none 无类型校验：truthy 归一为 True、falsy 归一为 False | 已覆盖（test_zero_grad_invalid） |
| 传参与不传参 | 显式 set_to_none=True 与默认不传参等价 | 已覆盖（test_zero_grad_default_equals_explicit_true） |
| 等价类/边界值 | set_to_none=False in-place 置 0 且引用保持、detach_/requires_grad_ 分支、view grad 报错、foreach 批量、sparse 逐张量置零 |
已覆盖（zero_grad_set_to_false/zero_grad_false_detach_semantics/zero_grad_foreach_batched_path/
zero_grad_sparse_grad_zeroed） |
| 正常传参场景 | 真实 backward 后清空可再次 backward 且不累积；多参数组多参数全量清空 | 已覆盖（zero_grad_then_backward_not_accumulated/
zero_grad_multiple_param_groups） |
| 异常传参场景 | zero_grad 无类型校验、无异常路径；set_to_none=False 遇 view grad 抛 RuntimeError（文档化 DDP 行为） | 已覆盖（zero_grad_invalid/
zero_grad_false_detach_semantics） |
| 混合设备类型 | NPU 与 CPU 清梯度后再 backward 的梯度一致（CPU Golden） | 已覆盖（test_zero_grad_cpu_golden_npu） |

未覆盖项及原因：
- foreach=True 批量置零路径在 NPU 上不支持 float64 梯度（aclnm _foreach_zero_ 仅支持 half/float/int/short/bfloat16），
  float64 用默认逐张量路径已覆盖；foreach+fp64 组合不在用例中断言，避免将 NPU 算子 dtype 限制固化为失败用例。
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
class TestOptimizerzero_grad(TestCase):
    """Test cases for torch.optim.Optimizer.zero_grad (NPU/CUDA portable).

    Covers all four set_to_none branches (explicit True / False / default /
    truthy-falsy normalization), the in-place zero + identity semantics of
    set_to_none=False (including detach_ and requires_grad_ branches and the
    documented view-grad RuntimeError), foreach batched zeroing, sparse
    per-tensor zeroing, re-backward after clearing, multi-group coverage,
    None-grad params untouched, and a CPU-vs-device golden comparison.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_opt(self, n=1, lr=0.01, **kwargs):
        params = [torch.nn.Parameter(torch.randn(3, device=self.device),
                                     requires_grad=True) for _ in range(n)]
        opt = torch.optim.SGD(params, lr=lr, **kwargs)
        for p in opt.param_groups[0]["params"]:
            p.grad = torch.ones_like(p)
        return opt

    # ------------------------------------------------------------------
    # set_to_none=True (explicit) and the default equivalence
    # ------------------------------------------------------------------

    def test_zero_grad_set_to_none(self):
        """Explicit set_to_none=True leaves grad None."""
        opt = self._make_opt()
        p = opt.param_groups[0]["params"][0]
        self.assertIsNotNone(p.grad, "precondition: grad must exist")
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad, "set_to_none=True must set grad to None")

    def test_zero_grad_default_equals_explicit_true(self):
        """The default call behaves exactly like set_to_none=True."""
        opt_default = self._make_opt()
        opt_explicit = self._make_opt()
        p_def = opt_default.param_groups[0]["params"][0]
        p_exp = opt_explicit.param_groups[0]["params"][0]
        opt_default.zero_grad()
        opt_explicit.zero_grad(set_to_none=True)
        self.assertIsNone(p_def.grad, "default call must clear the grad")
        self.assertIsNone(p_exp.grad, "explicit True must clear the grad")
        # Both cleared grads are None: identical post-state.
        self.assertEqual(p_def.grad, p_exp.grad)

    # ------------------------------------------------------------------
    # set_to_none=False: in-place zero with identity preserved
    # ------------------------------------------------------------------

    def test_zero_grad_set_to_false(self):
        """set_to_none=False zeros the tensor in place, keeping identity."""
        opt = self._make_opt()
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.ones_like(p)
        g = p.grad
        opt.zero_grad(set_to_none=False)
        self.assertIsNotNone(p.grad, "set_to_none=False must keep a Tensor")
        self.assertIs(p.grad, g,
                      "set_to_none=False must zero in place, not reallocate")
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))

    def test_zero_grad_false_detach_semantics(self):
        """set_to_none=False detaches non-leaf grads and clears
        requires_grad on leaf grads; view grads raise RuntimeError."""
        # Branch 1: a non-view non-leaf grad (has grad_fn) is detached in
        # place via detach_().
        base = torch.ones(3, device=self.device, requires_grad=True)
        p1 = torch.nn.Parameter(torch.ones(3, device=self.device))
        opt1 = torch.optim.SGD([p1], lr=0.1)
        p1.grad = base * 2  # fresh tensor with grad_fn (not a view)
        g1 = p1.grad
        self.assertIsNotNone(g1.grad_fn, "precondition: non-leaf grad")
        opt1.zero_grad(set_to_none=False)
        self.assertIs(p1.grad, g1, "identity must be preserved")
        self.assertIsNone(p1.grad.grad_fn,
                          "detach_() must strip the grad_fn in place")
        torch.testing.assert_close(p1.grad, torch.zeros_like(p1.grad))

        # Branch 2: a leaf grad with requires_grad=True has requires_grad
        # cleared in place via requires_grad_(False).
        p2 = torch.nn.Parameter(torch.ones(3, device=self.device))
        opt2 = torch.optim.SGD([p2], lr=0.1)
        p2.grad = torch.ones(3, device=self.device).requires_grad_(True)
        g2 = p2.grad
        self.assertIsNone(g2.grad_fn, "precondition: leaf grad")
        self.assertTrue(g2.requires_grad, "precondition: requires_grad=True")
        opt2.zero_grad(set_to_none=False)
        self.assertIs(p2.grad, g2, "identity must be preserved")
        self.assertFalse(p2.grad.requires_grad,
                         "requires_grad_ must be cleared in place")
        torch.testing.assert_close(p2.grad, torch.zeros_like(p2.grad))

        # Branch 3: a view grad cannot be detached in place; the documented
        # (DDP gradient_as_bucket_view) RuntimeError is raised, while
        # set_to_none=True handles the same view grad cleanly.
        p3 = torch.nn.Parameter(torch.ones(3, device=self.device))
        opt3 = torch.optim.SGD([p3], lr=0.1)
        p3.grad = (p3 * 2).sum().expand(3)
        with self.assertRaises(RuntimeError):
            opt3.zero_grad(set_to_none=False)
        p4 = torch.nn.Parameter(torch.ones(3, device=self.device))
        opt4 = torch.optim.SGD([p4], lr=0.1)
        p4.grad = (p4 * 2).sum().expand(3)
        opt4.zero_grad()
        self.assertIsNone(p4.grad,
                          "set_to_none=True must handle view grads cleanly")

    # ------------------------------------------------------------------
    # Truthy/falsy normalization (no type checking)
    # ------------------------------------------------------------------

    def test_zero_grad_invalid(self):
        """Truthy set_to_none behaves as True, falsy as False."""
        opt = self._make_opt()
        # zero_grad does not type-check set_to_none; any truthy value is
        # treated as True and zeroes to None, any falsy value as False.
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.ones_like(p)
        opt.zero_grad(set_to_none="yes")
        self.assertIsNone(p.grad, "truthy set_to_none must be treated as True")
        p.grad = torch.ones_like(p)
        opt.zero_grad(set_to_none=0)
        self.assertIsNotNone(p.grad, "falsy set_to_none must be treated as False")
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))

    # ------------------------------------------------------------------
    # Real backward: clearing enables a fresh, non-accumulated grad
    # ------------------------------------------------------------------

    def test_zero_grad_then_backward_not_accumulated(self):
        """After backward + zero_grad a new backward produces a fresh grad;
        without zero_grad in between the second backward accumulates."""
        def loss_fn(x):
            return ((x * 1.5) ** 2).sum()

        # Path 1: zero_grad between the two backwards -> fresh grad.
        x1 = torch.nn.Parameter(torch.ones(3, device=self.device))
        opt1 = torch.optim.SGD([x1], lr=0.1)
        loss_fn(x1).backward()
        first_grad = x1.grad.detach().clone()
        opt1.zero_grad()
        self.assertIsNone(x1.grad, "grad must be None after zero_grad")
        loss_fn(x1).backward()
        torch.testing.assert_close(x1.grad.detach(), first_grad,
                                   msg="re-backward after zero_grad must not accumulate")

        # Path 2 (control): no zero_grad in between -> grads accumulate.
        x2 = torch.nn.Parameter(torch.ones(3, device=self.device))
        opt2 = torch.optim.SGD([x2], lr=0.1)
        loss_fn(x2).backward()
        loss_fn(x2).backward()
        torch.testing.assert_close(x2.grad.detach(), 2 * first_grad,
                                   msg="without zero_grad the second backward must accumulate")

    # ------------------------------------------------------------------
    # Multiple param groups / multiple params
    # ------------------------------------------------------------------

    def test_zero_grad_multiple_param_groups(self):
        """All grads across multiple param groups and params are cleared."""
        pa = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        pb = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        pc = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt = torch.optim.SGD(
            [{"params": [pa, pb], "lr": 0.1}, {"params": [pc], "lr": 0.01}], lr=0.1)
        for p in (pa, pb, pc):
            p.grad = torch.ones_like(p)
        opt.zero_grad()
        self.assertTrue(all(p.grad is None for p in (pa, pb, pc)),
                        "set_to_none=True must clear every param in every group")

        # Same coverage for set_to_none=False with identity preservation.
        for p in (pa, pb, pc):
            p.grad = torch.ones_like(p)
        grads = [pa.grad, pb.grad, pc.grad]
        opt.zero_grad(set_to_none=False)
        for p, g in zip((pa, pb, pc), grads):
            self.assertIs(p.grad, g, "in-place identity must hold per param")
            torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))

    # ------------------------------------------------------------------
    # Params without grads are unaffected
    # ------------------------------------------------------------------

    def test_zero_grad_none_grad_unaffected(self):
        """Params whose grad is None stay None; only non-None grads change."""
        p_active = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        p_never = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        p_frozen = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=False)
        opt = torch.optim.SGD([p_active, p_never, p_frozen], lr=0.01)
        p_active.grad = torch.ones_like(p_active)
        opt.zero_grad()
        self.assertIsNone(p_active.grad, "the only real grad must be cleared")
        self.assertIsNone(p_never.grad, "a never-set grad must stay None")
        self.assertIsNone(p_frozen.grad, "a frozen param's grad must stay None")

        # set_to_none=False is also a no-op for None grads.
        p_active.grad = torch.ones_like(p_active)
        opt.zero_grad(set_to_none=False)
        self.assertIsNone(p_never.grad, "None grads must be untouched")
        torch.testing.assert_close(p_active.grad, torch.zeros_like(p_active.grad))

    # ------------------------------------------------------------------
    # foreach batched path and sparse per-tensor path
    # ------------------------------------------------------------------

    def test_zero_grad_foreach_batched_path(self):
        """foreach=True uses the batched _foreach_zero_ path with identity
        preserved for each grad."""
        p1 = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        p2 = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt = torch.optim.SGD([p1, p2], lr=0.1, foreach=True)
        p1.grad = torch.full_like(p1, 2.0)
        p2.grad = torch.full_like(p2, 3.0)
        g1, g2 = p1.grad, p2.grad
        opt.zero_grad(set_to_none=False)
        self.assertIs(p1.grad, g1)
        self.assertIs(p2.grad, g2)
        torch.testing.assert_close(p1.grad, torch.zeros_like(p1.grad))
        torch.testing.assert_close(p2.grad, torch.zeros_like(p2.grad))
        # The same optimizer with set_to_none=True drops the grads entirely.
        p1.grad = torch.full_like(p1, 2.0)
        p2.grad = torch.full_like(p2, 3.0)
        opt.zero_grad()
        self.assertIsNone(p1.grad)
        self.assertIsNone(p2.grad)

    def test_zero_grad_sparse_grad_zeroed(self):
        """A sparse grad bypasses the batched path and is zeroed per tensor,
        keeping its sparse layout."""
        try:
            p = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
            opt = torch.optim.SGD([p], lr=0.1, foreach=True)
            idx = torch.tensor([[0, 2]], device=self.device)
            p.grad = torch.sparse_coo_tensor(
                idx, torch.tensor([1.0, 2.0], device=self.device), (3,))
            self.assertTrue(p.grad.is_sparse, "precondition: sparse grad")
            opt.zero_grad(set_to_none=False)
            self.assertIsNotNone(p.grad)
            self.assertTrue(p.grad.is_sparse,
                            "zeroing must keep the sparse layout")
            torch.testing.assert_close(p.grad.to_dense().abs().sum().cpu(),
                                       torch.zeros(()),
                                       msg="sparse grad values must be zeroed")
        except (RuntimeError, ValueError) as exc:
            self.skipTest('sparse grad unsupported on this backend: %s' % exc)

    # ------------------------------------------------------------------
    # CPU golden comparison
    # ------------------------------------------------------------------

    def test_zero_grad_cpu_golden_npu(self):
        """After zero_grad a re-backward grad on device matches the CPU golden."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')

        def loss_fn(x):
            return ((x * 2.0) ** 2).sum()

        torch.manual_seed(0)
        cpu_x = torch.nn.Parameter(torch.randn(4))
        dev_x = torch.nn.Parameter(cpu_x.detach().clone().to(self.device))
        cpu_opt = torch.optim.SGD([cpu_x], lr=0.1)
        dev_opt = torch.optim.SGD([dev_x], lr=0.1)

        # First backward + zero on both sides.
        loss_fn(cpu_x).backward()
        loss_fn(dev_x).backward()
        cpu_opt.zero_grad(set_to_none=False)
        dev_opt.zero_grad(set_to_none=False)
        torch.testing.assert_close(dev_x.grad.cpu(), torch.zeros_like(cpu_x.grad),
                                   msg="device grads must be zeroed like CPU")
        # Re-backward after clearing: device grad must equal the CPU golden.
        loss_fn(cpu_x).backward()
        loss_fn(dev_x).backward()
        torch.testing.assert_close(dev_x.grad.detach().cpu(), cpu_x.grad.detach())


if __name__ == "__main__":
    run_tests()
