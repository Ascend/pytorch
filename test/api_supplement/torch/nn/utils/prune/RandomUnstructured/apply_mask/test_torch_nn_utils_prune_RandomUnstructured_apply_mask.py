# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.utils.prune.RandomUnstructured.apply_mask 接口功能正确性
API 名称：torch.nn.utils.prune.RandomUnstructured.apply_mask
API 签名：apply_mask(self, module: nn.Module) -> torch.Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | module 已含 weight_orig/weight_mask（apply 后的形态）与缺失 _orig 的未准备模块 | 已覆盖
（apply_mask_multiplies_mask_with_orig/apply_mask_without_orig_raises） |
| 枚举选项 | mask 与 orig 同 dtype 及 mask dtype 与 orig 不同（float16 mask + float32 orig） | 已覆盖
（apply_mask_multiplies_mask_with_orig/apply_mask_casts_mask_to_orig_dtype） |
| 参数类型 | RandomUnstructured 实例 + _tensor_name 绑定的 nn.Module | 已覆盖
（apply_mask_multiplies_mask_with_orig） |
| 传参与不传参 | 仅 module 单参固定签名 | 已覆盖（apply_mask_multiplies_mask_with_orig） |
| 等价类/边界值 | _tensor_name 为 None 时 AssertionError；每次调用返回新张量（非别名） | 已覆盖
（apply_mask_without_tensor_name_raises/apply_mask_returns_fresh_tensor） |
| 正常传参场景 | 返回 mask.to(orig.dtype)*orig，设备与 dtype 同 orig；__call__ 经 apply_mask 刷新
module[name] | 已覆盖（apply_mask_multiplies_mask_with_orig/
apply_mask_updates_module_weight_on_call） |
| 异常传参场景 | 未经 apply 准备的 module 缺 weight_mask 时 AttributeError（apply_mask 先取 mask 后取
orig）；_tensor_name 为 None 时 AssertionError | 已覆盖（apply_mask_without_orig_raises/
apply_mask_without_tensor_name_raises） |
| 混合设备类型 | 设备端乘积结果与 CPU 侧相同 mask/orig 的乘积一致（CPU Golden） | 已覆盖
（apply_mask_cpu_golden） |

未覆盖项及原因：
- apply_mask 不校验 mask/orig 形状一致性，形状不匹配输入的行为属上游广播语义，
  不构造非契约输入。

注意：apply_mask 的乘法在 NPU 设备张量上真实执行；Golden 为相同 mask/orig 在 CPU 上的
     乘积，浮点对照使用 torch.testing.assert_close。
"""
import os
import unittest

import torch

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

from torch.nn.utils.prune import BasePruningMethod, RandomUnstructured


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRandomUnstructuredApplyMask(TestCase):
    """Test cases for RandomUnstructured.apply_mask (NPU/CUDA portable).

    apply_mask() reads module[name_mask] and module[name_orig] and returns
    mask.to(orig.dtype) * orig as the pruned tensor; __call__ stores that
    result back into module[name].
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
        """A Linear pruned by RandomUnstructured.apply with its hook."""
        lin = self._seeded_linear()
        RandomUnstructured.apply(lin, 'weight', amount=3)
        hook = [h for h in lin._forward_pre_hooks.values()
                if isinstance(h, BasePruningMethod)][0]
        return lin, hook

    # ------------------------------------------------------------------
    # Core contract: mask.to(orig.dtype) * orig
    # ------------------------------------------------------------------

    def test_apply_mask_multiplies_mask_with_orig(self):
        """apply_mask returns the mask-orig product with orig dtype/device."""
        lin, hook = self._pruned_linear()
        pruned = hook.apply_mask(lin)
        expected = lin.weight_mask.to(lin.weight_orig.dtype) * lin.weight_orig
        self.assertEqual(pruned.dtype, lin.weight_orig.dtype,
                         'the pruned tensor must carry the original dtype')
        self.assertEqual(pruned.device, lin.weight_orig.device,
                         'the pruned tensor must stay on the original device')
        torch.testing.assert_close(pruned.detach().cpu(), expected.cpu(),
                                   msg='apply_mask must return mask * orig')

    def test_apply_mask_casts_mask_to_orig_dtype(self):
        """A lower-precision mask is cast to the original dtype before mul."""
        lin, hook = self._pruned_linear()
        lin.weight_mask = lin.weight_mask.to(torch.float16)
        pruned = hook.apply_mask(lin)
        self.assertEqual(pruned.dtype, torch.float32,
                         'a float16 mask must be cast to the orig dtype')
        expected = lin.weight_mask.to(torch.float32) * lin.weight_orig
        torch.testing.assert_close(pruned.detach().cpu(), expected.cpu(),
                                   msg='cast mask times orig must match the product')

    def test_apply_mask_returns_fresh_tensor(self):
        """Each apply_mask call yields a new tensor, not an alias."""
        lin, hook = self._pruned_linear()
        first = hook.apply_mask(lin)
        second = hook.apply_mask(lin)
        self.assertIsNot(first, second,
                         'apply_mask must produce a fresh tensor per call')

    def test_apply_mask_updates_module_weight_on_call(self):
        """__call__ re-applies apply_mask so module[name] tracks weight_orig."""
        lin, hook = self._pruned_linear()
        with torch.no_grad():
            lin.weight_orig.copy_(torch.randn_like(lin.weight_orig))
        hook(lin, None)  # the same entry point invoked as a forward pre-hook
        expected = lin.weight_mask.to(lin.weight_orig.dtype) * lin.weight_orig
        torch.testing.assert_close(lin.weight.detach().cpu(), expected.cpu(),
                                   msg='__call__ must refresh module weight via apply_mask')

    # ------------------------------------------------------------------
    # Exception paths
    # ------------------------------------------------------------------

    def test_apply_mask_without_orig_raises(self):
        """A module not prepared by apply() lacks weight_mask: AttributeError."""
        lin = self._seeded_linear()
        method = RandomUnstructured(amount=0.5)
        method._tensor_name = 'weight'
        # apply_mask fetches weight_mask first, then weight_orig; on a module
        # that was never pruned both are missing and the first getattr raises.
        with self.assertRaisesRegex(AttributeError, 'weight_mask'):
            method.apply_mask(lin)

    def test_apply_mask_without_tensor_name_raises(self):
        """A method without a bound tensor name raises AssertionError."""
        lin = self._seeded_linear()
        method = RandomUnstructured(amount=0.5)
        method._tensor_name = None
        with self.assertRaisesRegex(AssertionError, 'has to be pruned'):
            method.apply_mask(lin)

    # ------------------------------------------------------------------
    # CPU golden comparison
    # ------------------------------------------------------------------

    def test_apply_mask_cpu_golden(self):
        """Device product matches the CPU product for the same mask/orig."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        dev_lin, dev_hook = self._pruned_linear()
        cpu_lin = self._seeded_linear(device='cpu')
        cpu_lin.register_buffer('weight_mask', dev_lin.weight_mask.detach().cpu())
        cpu_lin.register_parameter(
            'weight_orig', torch.nn.Parameter(dev_lin.weight_orig.detach().cpu()))
        method = RandomUnstructured(amount=3)
        method._tensor_name = 'weight'
        pruned_dev = dev_hook.apply_mask(dev_lin)
        pruned_cpu = method.apply_mask(cpu_lin)
        torch.testing.assert_close(pruned_dev.detach().cpu(), pruned_cpu.detach(),
                                   msg='device mask product must match the CPU golden')


if __name__ == "__main__":
    run_tests()
