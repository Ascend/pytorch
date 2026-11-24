# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.utils.prune.RandomUnstructured.remove 接口功能正确性
API 名称：torch.nn.utils.prune.RandomUnstructured.remove
API 签名：remove(self, module: nn.Module) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 已剪枝模块（存在 _orig/_mask）与未剪枝模块两种输入形态 | 已覆盖
（remove_keeps_pruned_weights/remove_not_pruned_raises） |
| 枚举选项 | 经 apply 后的 forward pre-hook 实例直接调用 remove 与公共入口
torch.nn.utils.prune.remove 两种调用形态 | 已覆盖（remove_via_method_instance/
remove_public_entry_delegates_to_method） |
| 参数类型 | RandomUnstructured（继承 BasePruningMethod.remove）+ nn.Module | 已覆盖
（remove_keeps_pruned_weights） |
| 传参与不传参 | 仅 module 单参固定签名 | 已覆盖（remove_keeps_pruned_weights） |
| 等价类/边界值 | 剪枝不可逆：移除后置零保持；weight_orig 数据合并回 weight；方法级 remove 不注销
hook、公共入口 prune.remove 注销 hook | 已覆盖（remove_keeps_pruned_weights/
remove_public_entry_delegates_to_method/remove_training_continues_after_remove） |
| 正常传参场景 | 移除后 weight 为普通 Parameter、_orig/_mask 全部移除、前向结果不变 | 已覆盖
（remove_keeps_pruned_weights/remove_forward_still_works/
remove_cpu_golden） |
| 异常传参场景 | 未剪枝模块：方法实例 _tensor_name=None 时 AssertionError；公共入口
prune.remove 对未剪枝模块 ValueError | 已覆盖（remove_not_pruned_raises/
remove_public_entry_delegates_to_method） |
| 混合设备类型 | 设备端移除后置零个数与 CPU Golden 一致（同为精确 amount） | 已覆盖
（remove_cpu_golden） |

未覆盖项及原因：
- PruningContainer.remove（多方法组合）不属本 API（RandomUnstructured.remove），
  组合移除行为由 PruningContainer 自身看护。

注意：remove 在 NPU 设备模块上真实执行（apply_mask 乘法与参数重注册走设备算子）；
     Golden 为 CPU 侧同种子同 amount 的移除结果（置零个数），浮点对照使用
     torch.testing.assert_close。
"""
import os
import unittest

import torch
import torch.nn.functional as F

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe the device only when the supplement tests actually run: plain CI
# collection keeps the class-level skipUnless gate and skips device init.
DEVICE_NAME = 'cpu'
if RUN_FLAG:
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

from torch.nn.utils import prune as prune_module
from torch.nn.utils.prune import BasePruningMethod, RandomUnstructured


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRandomUnstructuredRemove(TestCase):
    """Test cases for RandomUnstructured.remove (NPU/CUDA portable).

    remove() makes pruning permanent: the masked weight becomes a plain
    parameter, weight_orig/weight_mask and the pruning hook are removed, and
    previously pruned units stay zero.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _seeded_linear(self, device=None):
        if device is None:
            device = self.device_name
        torch.manual_seed(42)
        cpu_lin = torch.nn.Linear(10, 4)
        lin = torch.nn.Linear(10, 4).to(device)
        lin.load_state_dict(cpu_lin.state_dict())
        return lin

    def _pruned_linear(self):
        lin = self._seeded_linear()
        RandomUnstructured.apply(lin, 'weight', amount=3)
        hook = [h for h in lin._forward_pre_hooks.values()
                if isinstance(h, BasePruningMethod)][0]
        return lin, hook

    # ------------------------------------------------------------------
    # Removal contract: pruning is permanent, reparameterization is gone
    # ------------------------------------------------------------------

    def test_remove_keeps_pruned_weights(self):
        """remove() keeps masked values and drops _orig/_mask; hook stays.

        The method-level remove() only strips the reparameterization; the
        registered forward pre-hook stays until the public wrapper
        torch.nn.utils.prune.remove deletes it (covered below).
        """
        lin, _ = self._pruned_linear()
        masked = lin.weight.detach().clone()
        pruned_units = int((masked == 0).sum())
        self.assertEqual(pruned_units, 3, 'precondition: exact pruned unit count')
        hook = [h for h in lin._forward_pre_hooks.values()
                if isinstance(h, BasePruningMethod)][0]
        hook.remove(lin)
        self.assertNotIn('weight_orig', dict(lin.named_parameters()),
                         'weight_orig must be removed')
        self.assertNotIn('weight_mask', dict(lin.named_buffers()),
                         'weight_mask must be removed')
        torch.testing.assert_close(lin.weight.detach().cpu(), masked.cpu(),
                                   msg='the weight must keep the pruned (masked) values')
        self.assertEqual(int((lin.weight.detach() == 0).sum()), 3,
                         'pruning must stay permanent after remove')
        self.assertIsInstance(lin.weight, torch.nn.Parameter,
                              'weight must be a plain Parameter again')
        self.assertTrue(lin.weight.requires_grad, 'weight must stay trainable')
        remaining = [h for h in lin._forward_pre_hooks.values()
                     if isinstance(h, BasePruningMethod)]
        self.assertEqual(remaining, [hook],
                         'the method-level remove does not unregister the hook')

    def test_remove_via_method_instance(self):
        """RandomUnstructured.remove called on the hook instance directly."""
        lin, hook = self._pruned_linear()
        hook.remove(lin)  # the RandomUnstructured.remove entry point itself
        self.assertNotIn('weight_mask', dict(lin.named_buffers()))
        self.assertNotIn('weight_orig', dict(lin.named_parameters()))

    def test_remove_public_entry_delegates_to_method(self):
        """torch.nn.utils.prune.remove finds the hook, calls remove, unregisters."""
        lin, _ = self._pruned_linear()
        masked = lin.weight.detach().clone()
        returned = prune_module.remove(lin, 'weight')
        self.assertIs(returned, lin)
        torch.testing.assert_close(lin.weight.detach().cpu(), masked.cpu(),
                                   msg='the public entry must keep the masked values')
        self.assertNotIn('weight_mask', dict(lin.named_buffers()))
        remaining = [h for h in lin._forward_pre_hooks.values()
                     if isinstance(h, BasePruningMethod)]
        self.assertEqual(remaining, [],
                         'the public entry must unregister the pruning hook')
        # The public entry raises ValueError on a module that was never pruned.
        with self.assertRaisesRegex(ValueError, 'has to be pruned'):
            prune_module.remove(self._seeded_linear(), 'weight')

    def test_remove_forward_still_works(self):
        """After remove the module forward uses the permanent masked weight."""
        lin, _ = self._pruned_linear()
        weight_after = lin.weight.detach().clone()
        prune_module.remove(lin, 'weight')
        x = torch.randn(2, 10).to(self.device_name)
        with torch.no_grad():
            out = lin(x)
            expected = F.linear(x, weight_after, lin.bias)
        if self.device_name == 'npu':
            torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), expected.cpu(),
                                   msg='forward after remove must use the masked weight')

    def test_remove_training_continues_after_remove(self):
        """Training after remove updates the plain parameter without re-masking."""
        lin, _ = self._pruned_linear()
        prune_module.remove(lin, 'weight')
        weight_before = lin.weight.detach().clone()
        loss = lin(torch.randn(3, 10).to(self.device_name)).sum()
        loss.backward()
        optimizer = torch.optim.SGD(lin.parameters(), lr=0.1)
        optimizer.step()
        if self.device_name == 'npu':
            torch.npu.synchronize()
        self.assertFalse(torch.allclose(lin.weight.detach().cpu(), weight_before.cpu()),
                         'the plain weight must keep updating after remove')
        self.assertTrue(lin.weight.is_leaf,
                        'weight must be a plain leaf parameter after remove')

    # ------------------------------------------------------------------
    # Exception path
    # ------------------------------------------------------------------

    def test_remove_not_pruned_raises(self):
        """A method instance without a pruned tensor raises AssertionError."""
        lin = self._seeded_linear()
        method = RandomUnstructured(amount=0.5)
        method._tensor_name = None
        with self.assertRaisesRegex(AssertionError, 'has to be pruned'):
            method.remove(lin)

    # ------------------------------------------------------------------
    # CPU golden comparison
    # ------------------------------------------------------------------

    def test_remove_cpu_golden(self):
        """Device removal keeps the same exact pruned count as the CPU golden.

        Also covers the documented refresh contract: remove() applies the
        current mask to the current weight_orig, so updates made between
        apply() and remove() are reflected in the final weight.
        """
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        dev_lin, _ = self._pruned_linear()
        torch.manual_seed(7)
        with torch.no_grad():
            dev_lin.weight_orig.copy_(torch.randn_like(dev_lin.weight_orig))
        mask_dev = dev_lin.weight_mask.detach().clone()
        orig_dev = dev_lin.weight_orig.detach().clone()
        prune_module.remove(dev_lin, 'weight')
        self.assertNotIn('weight_mask', dict(dev_lin.named_buffers()))
        torch.testing.assert_close(dev_lin.weight.detach().cpu(),
                                   mask_dev.cpu() * orig_dev.cpu(),
                                   msg='remove must bake the updated orig and the mask '
                                       'into the final weight')
        self.assertEqual(int((dev_lin.weight.detach() == 0).sum()), 3,
                         'the refreshed weight must keep the pruned units zero')
        cpu_lin = self._seeded_linear(device='cpu')
        RandomUnstructured.apply(cpu_lin, 'weight', amount=3)
        prune_module.remove(cpu_lin, 'weight')
        self.assertEqual(int((dev_lin.weight.detach() == 0).sum()),
                         int((cpu_lin.weight.detach() == 0).sum()),
                         'device and CPU removals must keep the same pruned count')


if __name__ == "__main__":
    run_tests()
