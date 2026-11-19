# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.AdaptiveLogSoftmaxWithLoss.reset_parameters 接口功能正确性
API 名称：torch.nn.AdaptiveLogSoftmaxWithLoss.reset_parameters
API 签名：reset_parameters(self) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | head 与全部 tail 子模块参数均被重置；head_bias=True/False 两种结构 | 已覆盖（test_reset_parameters_affects_head_and_tail） |
| 枚举选项 | head_bias 开关两分支均参与精确对照 | 已覆盖（matches_fresh_construction/head_bias_false） |
| 参数类型 | 重置前后参数形状、dtype、数量、设备保持不变 | 已覆盖（test_reset_parameters_preserves_param_objects） |
| 传参与不传参 | 无参方法，仅默认调用 | 已覆盖（test_reset_parameters_matches_fresh_construction） |
| 等价类/边界值 | 重置值精确等于同 RNG 状态全新模块的初始化值（上界 1/sqrt(fan_in)） | 已覆盖（matches_fresh_construction/bounds_on_device） |
| 正常传参场景 | 设备侧扰动后重置恢复初始化分布；重置后前向可用 | 已覆盖（test_reset_parameters_on_device/test_reset_parameters_forward_after_reset） |
| 异常传参场景 | 无参方法无校验路径 | 未覆盖（reset_parameters 无入参校验，无稳定异常契约） |
| 混合设备类型 | CPU 精确对照 + 设备侧（NPU/CUDA）重置与 RNG 续流精确对照 | 已覆盖（test_reset_parameters_device_continuation） |

未覆盖项及原因：
- 异常传参场景不适用：reset_parameters 无参数、无校验分支。
- 本文件同时看护别名路径 torch.nn.modules.adaptive.AdaptiveLogSoftmaxWithLoss.reset_parameters：
  torch.nn.AdaptiveLogSoftmaxWithLoss 与 torch.nn.modules.adaptive.AdaptiveLogSoftmaxWithLoss
  为同一类对象，见 identity 断言用例。

注意：重置语义验证使用 RNG 续流精确对照——同种子下 A/B/C 三模块，A、B 连续构造消耗
     两段随机数，C 重置后必须与 B 精确一致（torch.equal），reset 若跳过或乱序任何一次
     初始化抽样该断言即失败；设备侧浮点值按确定边界（1/sqrt(fan_in)）断言，不做
     CPU/NPU 数值对照（两侧随机数生成器不同）。
"""
import math
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"

DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


def _asm_kwargs():
    """Common module configuration for all AdaptiveLogSoftmaxWithLoss cases."""
    return dict(in_features=6, n_classes=20, cutoffs=[4, 9], div_value=2.0,
                head_bias=True)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestAdaptiveLogSoftmaxWithLossResetParameters(TestCase):
    """Comprehensive cases for AdaptiveLogSoftmaxWithLoss.reset_parameters (portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn and torch.nn.modules.adaptive expose the same ASM class."""
        self.assertIs(
            torch.nn.AdaptiveLogSoftmaxWithLoss,
            torch.nn.modules.adaptive.AdaptiveLogSoftmaxWithLoss)

    def _module(self, **overrides):
        """Build an AdaptiveLogSoftmaxWithLoss on the current device."""
        kwargs = _asm_kwargs()
        kwargs.update(overrides)
        return torch.nn.AdaptiveLogSoftmaxWithLoss(device=self.device, **kwargs)

    def test_reset_parameters_matches_fresh_construction(self):
        """Reset reproduces exactly what a fresh module draws next (RNG continuation)."""
        torch.manual_seed(17)
        first_draw = self._module()  # A: consumes D1
        reference = self._module()  # B: consumes D2
        torch.manual_seed(17)
        target = self._module()  # C: consumes D1 again
        self.assertTrue(torch.equal(target.head.weight, first_draw.head.weight))
        self.assertFalse(torch.equal(target.head.weight, reference.head.weight))
        target.reset_parameters()
        for param_target, param_reference in zip(target.parameters(),
                                                 reference.parameters()):
            self.assertTrue(torch.equal(param_target, param_reference))

    def test_reset_parameters_deterministic_under_seed(self):
        """Two fresh constructions under the same seed produce identical params."""
        torch.manual_seed(29)
        first = self._module()
        torch.manual_seed(29)
        second = self._module()
        for param_first, param_second in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(param_first, param_second))

    def test_reset_parameters_affects_head_and_tail(self):
        """Reset restores both the head and every tail projection after perturbation."""
        target = self._module()
        with torch.no_grad():
            for param in target.parameters():
                param.fill_(77.0)
        target.reset_parameters()
        for name, param in target.named_parameters():
            self.assertFalse(bool((param == 77.0).all()),
                             f"{name} was not reset")
        self.assertFalse(bool((target.head.weight == 77.0).all()))
        for projection in target.tail:
            self.assertFalse(bool((projection[0].weight == 77.0).all()))
            self.assertFalse(bool((projection[1].weight == 77.0).all()))

    def test_reset_parameters_head_bias_false(self):
        """head_bias=False configuration resets cleanly with no head bias draw."""
        torch.manual_seed(23)
        torch.nn.AdaptiveLogSoftmaxWithLoss(
            in_features=6, n_classes=20, cutoffs=[4, 9], div_value=2.0,
            head_bias=False)  # A: consumes D1
        reference = torch.nn.AdaptiveLogSoftmaxWithLoss(
            in_features=6, n_classes=20, cutoffs=[4, 9], div_value=2.0,
            head_bias=False)  # B: consumes D2
        torch.manual_seed(23)
        target = torch.nn.AdaptiveLogSoftmaxWithLoss(
            in_features=6, n_classes=20, cutoffs=[4, 9], div_value=2.0,
            head_bias=False)  # C: consumes D1 again
        target.reset_parameters()
        for param_target, param_reference in zip(target.parameters(),
                                                 reference.parameters()):
            self.assertTrue(torch.equal(param_target, param_reference))

    def test_reset_parameters_bounds_on_device(self):
        """After perturbation on device, reset restores the Linear init bounds."""
        target = self._module()
        with torch.no_grad():
            for param in target.parameters():
                param.fill_(77.0)
        target.reset_parameters()
        weights = {name: param for name, param in target.named_parameters()
                   if name.endswith(".weight")}
        for name, param in target.named_parameters():
            fan_in = weights[name[:-len(".bias")] + ".weight"].shape[1] \
                if name.endswith(".bias") else param.shape[1]
            bound = 1.0 / math.sqrt(fan_in)
            self.assertTrue(bool((param.abs() <= bound).all()),
                            f"{name} exceeds the init bound")
            self.assertEqual(param.device.type, self.device_name)

    def test_reset_parameters_preserves_param_objects(self):
        """Reset keeps the same Parameter objects, shapes and dtypes."""
        target = self._module()
        refs = list(target.parameters())
        shapes = [tuple(p.shape) for p in refs]
        target.reset_parameters()
        for ref, current in zip(refs, target.parameters()):
            self.assertIs(current, ref)
        self.assertEqual([tuple(p.shape) for p in target.parameters()], shapes)
        self.assertTrue(all(p.dtype == torch.float32 for p in target.parameters()))

    def test_reset_parameters_device_continuation(self):
        """RNG continuation equality also holds for device-allocated parameters."""
        torch.manual_seed(19)
        self._module()  # A: consumes device draws D1
        reference = self._module()  # B: consumes D2
        torch.manual_seed(19)
        target = self._module()  # C: consumes D1 again
        target.reset_parameters()
        for param_target, param_reference in zip(target.parameters(),
                                                 reference.parameters()):
            self.assertTrue(torch.equal(param_target, param_reference))
        self.assertEqual(reference.head.weight.device.type, self.device_name)

    def test_reset_parameters_forward_after_reset(self):
        """Forward after reset returns the documented NamedTuple structure."""
        target = self._module()
        target.reset_parameters()
        x = torch.randn(4, 6, device=self.device)
        targets = torch.tensor([1, 5, 9, 15], device=self.device)
        result = target(x, targets)
        self.assertEqual(tuple(result.output.shape), (4,))
        self.assertEqual(result.loss.dim(), 0)
        log_prob = target.log_prob(x)
        self.assertEqual(tuple(log_prob.shape), (4, 20))


if __name__ == "__main__":
    run_tests()
