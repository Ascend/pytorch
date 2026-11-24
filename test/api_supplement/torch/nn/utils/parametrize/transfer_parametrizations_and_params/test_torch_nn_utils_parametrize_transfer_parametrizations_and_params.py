# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.utils.parametrize.transfer_parametrizations_and_params 接口功能正确性
API 名称：torch.nn.utils.parametrize.transfer_parametrizations_and_params
API 签名：transfer_parametrizations_and_params(from_module, to_module,
         tensor_name: str | None = None) -> Module

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | from_module 已参数化/未参数化（未参数化时直接返回 to_module，无副作用） | 已覆盖
（transfer_parametrizations_and_params_transfers/
transfer_parametrizations_and_params_not_parametrized_noop） |
| 枚举选项 | tensor_name=None（转移全部参数化）与 tensor_name='weight'（仅转移指定参数：
源上第二属性 bias 的参数化不被转移） | 已覆盖
（tensor_name=None：transfers/shares_original/device_golden，多参数化源上的
"全部转移"由 transfers_all_multiple 看护；
tensor_name='weight'：named_tensor） |
| 参数类型 | 自定义 parametrization（Skew）；to_module 已有/没有同名参数两种目标形态 | 已覆盖
（transfer_parametrizations_and_params_transfers/
transfer_parametrizations_and_params_missing_param_created） |
| 传参与不传参 | tensor_name 省略（默认 None，转移全部）与显式传入 'weight' 为两条独立路径：
省略路径由 transfers 覆盖（多参数化源上省略 tensor_name 转移全部由
transfers_all_multiple 看护），显式定向性（未参数化目标上仅转移指定参数）由
named_tensor 覆盖 | 已覆盖
（transfer_parametrizations_and_params_transfers/
transfer_parametrizations_and_params_transfers_all_multiple/
transfer_parametrizations_and_params_named_tensor） |
| 等价类/边界值 | 返回 to_module 同对象；to_module 的 original 与 from_module 共享（改 from 的 original
会同步影响 to 的权重） | 已覆盖（transfer_parametrizations_and_params_shares_original） |
| 正常传参场景 | 转移后 to_module 权重与 from_module 权重一致且参数化已注册 | 已覆盖
（transfer_parametrizations_and_params_transfers/
transfer_parametrizations_and_params_device_golden） |
| 异常传参场景 | from_module 未参数化时按文档 does nothing（非异常路径，作为行为契约用例） | 已覆盖
（transfer_parametrizations_and_params_not_parametrized_noop） |
| 混合设备类型 | 相同种子的设备端转移结果与 CPU Golden 一致 | 已覆盖
（transfer_parametrizations_and_params_device_golden） |

未覆盖项及原因：
- from_module.parametrizations 非 ModuleDict、parameters_to_transfer 不可迭代的
  AssertionError 分支为内部防御断言，正常注册路径下不可构造，未覆盖。
- 同一参数上叠加多个 parametrization（original0/original1 多原始张量命名）未单独构造；
  本文件的多参数化场景为不同属性各挂一个 parametrization（named_tensor 与
  transfers_all_multiple 用例中 from_module 的 weight 与 bias），单属性多参数化
  叠加分支行为一致，未单独构造。

注意：转移在 NPU 设备模块上真实执行；Golden 为相同种子下 CPU 侧转移结果，
     浮点对照使用 torch.testing.assert_close。
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

from torch.nn.utils.parametrize import (
    is_parametrized,
    register_parametrization,
    transfer_parametrizations_and_params,
)


class Skew(torch.nn.Module):
    """Map an arbitrary square matrix to a skew-symmetric one."""

    def forward(self, x):
        upper = x.triu(1)
        return upper - upper.transpose(-1, -2)


class Passthrough(torch.nn.Module):
    """Leave the original tensor unchanged; marks an attribute as parametrized."""

    def forward(self, x):
        return x


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTransferParametrizationsAndParams(TestCase):
    """Test cases for transfer_parametrizations_and_params (NPU/CUDA portable).

    Transferring registers the parametrizations of ``from_module`` on
    ``to_module`` (every parametrized parameter by default, or only the
    ``tensor_name``-selected one) and points the new ``original`` tensors at
    the source ones, so both modules derive identical parametrized values
    from shared state.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _parametrized_linear(self, device=None):
        if device is None:
            device = self.device_name
        torch.manual_seed(42)
        module = torch.nn.Linear(4, 4).to(device)
        register_parametrization(module, 'weight', Skew())
        module.weight  # materialize the parametrized value
        return module

    # ------------------------------------------------------------------
    # Functional contract
    # ------------------------------------------------------------------

    def test_transfer_parametrizations_and_params_transfers(self):
        """Transfer registers the parametrization and yields equal weights."""
        from_module = self._parametrized_linear()
        to_module = self._parametrized_linear()
        returned = transfer_parametrizations_and_params(from_module, to_module)
        self.assertIs(returned, to_module, 'to_module must be returned unchanged')
        self.assertTrue(is_parametrized(to_module),
                        'to_module must be parametrized after the transfer')
        q_from = from_module.weight.detach()
        q_to = to_module.weight.detach()
        if self.device_name == 'npu':
            torch.npu.synchronize()
        torch.testing.assert_close(q_to.cpu(), q_from.cpu(),
                                   msg='transferred weight must equal the source weight')
        self.assertIs(to_module.parametrizations.weight.original,
                      from_module.parametrizations.weight.original,
                      'the transfer must point to_module.original at the source original')

    def test_transfer_parametrizations_and_params_transfers_all_multiple(self):
        """tensor_name=None transfers every parametrized attribute of the source.

        The source carries two parametrized parameters (weight via Skew and
        bias via Passthrough) and the target starts unparametrized (asserted
        as a precondition), so omitting tensor_name must transfer both
        parametrizations, share both originals with the source and keep both
        transferred values equal to the source ones.
        """
        from_module = self._parametrized_linear()
        register_parametrization(from_module, 'bias', Passthrough())
        from_module.bias  # materialize the parametrized value
        to_module = torch.nn.Linear(4, 4).to(self.device_name)
        self.assertFalse(is_parametrized(to_module),
                         'precondition: the target must not be parametrized yet')
        self.assertTrue(is_parametrized(from_module, 'weight'),
                        'precondition: the source weight must be parametrized')
        self.assertTrue(is_parametrized(from_module, 'bias'),
                        'precondition: the source bias must be parametrized')
        returned = transfer_parametrizations_and_params(from_module, to_module)
        self.assertIs(returned, to_module, 'to_module must be returned unchanged')
        self.assertTrue(is_parametrized(to_module, 'weight'),
                        'the default transfer must register the weight parametrization')
        self.assertTrue(is_parametrized(to_module, 'bias'),
                        'the default transfer must register the bias parametrization')
        self.assertIs(to_module.parametrizations.weight.original,
                      from_module.parametrizations.weight.original,
                      'the transferred weight original must be shared with the source')
        self.assertIs(to_module.parametrizations.bias.original,
                      from_module.parametrizations.bias.original,
                      'the transferred bias original must be shared with the source')
        q_weight_from = from_module.weight.detach()
        q_weight_to = to_module.weight.detach()
        q_bias_from = from_module.bias.detach()
        q_bias_to = to_module.bias.detach()
        if self.device_name == 'npu':
            torch.npu.synchronize()
        torch.testing.assert_close(q_weight_to.cpu(), q_weight_from.cpu(),
                                   msg='transferred weight must equal the source weight')
        torch.testing.assert_close(q_bias_to.cpu(), q_bias_from.cpu(),
                                   msg='transferred bias must equal the source bias')

    def test_transfer_parametrizations_and_params_named_tensor(self):
        """tensor_name='weight' transfers only the requested parameter.

        The target starts unparametrized (asserted as a precondition) so the
        checks fail if the named transfer stops taking effect, and the source
        carries a second parametrized attribute (bias) that must not be
        transferred along with the named one.
        """
        from_module = self._parametrized_linear()
        register_parametrization(from_module, 'bias', Passthrough())
        to_module = torch.nn.Linear(4, 4).to(self.device_name)
        self.assertFalse(is_parametrized(to_module),
                         'precondition: the target must not be parametrized yet')
        self.assertTrue(is_parametrized(from_module, 'bias'),
                        'precondition: the source must carry a bias parametrization')
        returned = transfer_parametrizations_and_params(
            from_module, to_module, tensor_name='weight')
        self.assertIs(returned, to_module)
        self.assertTrue(is_parametrized(to_module, 'weight'),
                        'the named transfer must register the weight parametrization')
        self.assertFalse(is_parametrized(to_module, 'bias'),
                         "tensor_name='weight' must not transfer the bias parametrization")
        q_from = from_module.weight.detach()
        q_to = to_module.weight.detach()
        if self.device_name == 'npu':
            torch.npu.synchronize()
        torch.testing.assert_close(q_to.cpu(), q_from.cpu(),
                                   msg='named transfer must equal the source weight')

    def test_transfer_parametrizations_and_params_shares_original(self):
        """Mutating the source original is visible through the target weight."""
        from_module = self._parametrized_linear()
        to_module = self._parametrized_linear()
        transfer_parametrizations_and_params(from_module, to_module)
        with torch.no_grad():
            from_module.parametrizations.weight.original.add_(1.0)
        q_from = from_module.weight.detach()
        q_to = to_module.weight.detach()
        if self.device_name == 'npu':
            torch.npu.synchronize()
        torch.testing.assert_close(q_to.cpu(), q_from.cpu(),
                                   msg='both modules must read the shared original')

    def test_transfer_parametrizations_and_params_missing_param_created(self):
        """A target without the parameter gets one created and parametrized."""
        from_module = self._parametrized_linear()
        to_module = torch.nn.Module().to(self.device_name)
        transfer_parametrizations_and_params(from_module, to_module)
        self.assertTrue(hasattr(to_module, 'weight'),
                        'a missing parameter must be created on to_module')
        self.assertTrue(is_parametrized(to_module))
        q_from = from_module.weight.detach()
        q_to = to_module.weight.detach()
        if self.device_name == 'npu':
            torch.npu.synchronize()
        torch.testing.assert_close(q_to.cpu(), q_from.cpu(),
                                   msg='created parameter must produce the source weight')

    def test_transfer_parametrizations_and_params_not_parametrized_noop(self):
        """An unparametrized source is a documented no-op on to_module."""
        to_module = self._parametrized_linear()
        plain = torch.nn.Linear(4, 4).to(self.device_name)
        returned = transfer_parametrizations_and_params(plain, to_module)
        self.assertIs(returned, to_module, 'to_module must still be returned')
        self.assertTrue(is_parametrized(to_module),
                        'an existing target parametrization must stay intact')
        plain_target = torch.nn.Linear(4, 4).to(self.device_name)
        transfer_parametrizations_and_params(plain, plain_target)
        self.assertFalse(is_parametrized(plain_target),
                         'transferring from an unparametrized module does nothing')

    # ------------------------------------------------------------------
    # CPU golden comparison
    # ------------------------------------------------------------------

    def test_transfer_parametrizations_and_params_device_golden(self):
        """Same-seeded device transfer matches the CPU transferred weight."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        from_cpu = self._parametrized_linear(device='cpu')
        to_cpu = self._parametrized_linear(device='cpu')
        transfer_parametrizations_and_params(from_cpu, to_cpu)
        from_dev = self._parametrized_linear(device=self.device_name)
        to_dev = self._parametrized_linear(device=self.device_name)
        transfer_parametrizations_and_params(from_dev, to_dev)
        torch.testing.assert_close(to_dev.weight.detach().cpu(),
                                   to_cpu.weight.detach(),
                                   msg='device transferred weight must match the CPU golden')


if __name__ == "__main__":
    run_tests()
