# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.utils.fuse_linear_bn_eval 接口功能正确性
API 名称：torch.nn.utils.fuse_linear_bn_eval
API 签名：fuse_linear_bn_eval(linear: torch.nn.Linear,
         bn: torch.nn.modules.batchnorm._BatchNorm) -> torch.nn.Linear

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | linear 有/无 bias（bias=None 走 zeros 融合路径） | 已覆盖（fuse_linear_bn_eval_forward_matches_
original_chain/fuse_linear_bn_eval_no_bias_linear） |
| 枚举选项 | bn.num_features == linear.out_features 与 bn.num_features == 1（广播）两条合法路径 | 已覆盖
（fuse_linear_bn_eval_forward_matches_original_chain/
fuse_linear_bn_eval_num_features_one_broadcast） |
| 参数类型 | torch.nn.Linear + BatchNorm1d（eval 模式） | 已覆盖（fuse_linear_bn_eval_forward_matches_
original_chain） |
| 传参与不传参 | 仅两参固定签名，无可选参数 | 已覆盖（fuse_linear_bn_eval_forward_matches_original_chain） |
| 等价类/边界值 | 融合前输入模块参数不被改写、返回新模块（deepcopy 语义）、running_mean/var 参与融合 | 已覆盖
（fuse_linear_bn_eval_input_modules_unchanged） |
| 正常传参场景 | 融合后 Linear 前向等于原 linear→bn 链路；融合权重等于 scale 公式 Golden | 已覆盖
（fuse_linear_bn_eval_forward_matches_original_chain/fuse_linear_bn_eval_fused_weight_formula/
fuse_linear_bn_eval_cpu_golden） |
| 异常传参场景 | linear 或 bn 处于 training 时 AssertionError；out_features 与 num_features 不匹配
（且非 1）时 AssertionError | 已覆盖（fuse_linear_bn_eval_training_mode_raises/
fuse_linear_bn_eval_num_features_mismatch_raises） |
| 混合设备类型 | 相同种子的设备端融合权重/偏置与 CPU Golden 一致 | 已覆盖（fuse_linear_bn_eval_cpu_golden） |

未覆盖项及原因：
- bn.running_mean 为 None 的 AssertionError 分支不可达：LazyBatchNorm1d 物化前
  num_features 为 None，会先命中 num_features 校验分支，无法构造 running_mean 为 None
  且 num_features 匹配的输入。
- BatchNorm2d/3d 入参在 torch.nn.utils.fusion.fuse_linear_bn_eval 的用例中覆盖（同函数），
  本文件使用 BatchNorm1d 与 Linear 输出的典型组合。

注意：融合为纯权重运算，但被融合模块与输入张量均置于 NPU 设备上执行；
     融合输出与原 linear→bn 链路输出、融合权重与 CPU 公式 Golden 的浮点对照均使用
     torch.testing.assert_close（float32 默认容差）。
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


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestFuseLinearBnEval(TestCase):
    """Test cases for torch.nn.utils.fuse_linear_bn_eval (NPU/CUDA portable).

    The function fuses an eval-mode Linear and eval-mode BatchNorm into a new
    Linear whose weight/bias implement the same affine transform. Golden
    values follow fuse_linear_bn_weights: scale = bn_w * rsqrt(running_var +
    eps), fused_weight = weight * scale, fused_bias = (bias - running_mean) *
    scale + bn_b.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _seeded_pair(self, in_features=4, out_features=8, bias=True, device=None):
        """Same-seeded Linear + BatchNorm1d pair moved to ``device``."""
        if device is None:
            device = self.device_name
        torch.manual_seed(42)
        cpu_lin = torch.nn.Linear(in_features, out_features, bias=bias).eval()
        cpu_bn = torch.nn.BatchNorm1d(out_features).eval()
        with torch.no_grad():
            cpu_bn.weight.copy_(torch.randn(out_features))
            cpu_bn.bias.copy_(torch.randn(out_features))
            cpu_bn.running_mean.copy_(torch.randn(out_features))
            cpu_bn.running_var.copy_(torch.rand(out_features) + 0.2)
        lin = torch.nn.Linear(in_features, out_features, bias=bias).eval().to(device)
        bn = torch.nn.BatchNorm1d(out_features).eval().to(device)
        lin.load_state_dict(cpu_lin.state_dict())
        bn.load_state_dict(cpu_bn.state_dict())
        return lin, bn, cpu_lin, cpu_bn

    # ------------------------------------------------------------------
    # Functional contract: fused forward equals the original chain
    # ------------------------------------------------------------------

    def test_fuse_linear_bn_eval_forward_matches_original_chain(self):
        """Fused Linear forward equals linear->bn on the same device input."""
        lin, bn, _, _ = self._seeded_pair()
        x = torch.randn(3, 4)
        fused = torch.nn.utils.fuse_linear_bn_eval(lin, bn)
        self.assertIs(fused.training, False, 'fused module must inherit eval mode')
        with torch.no_grad():
            out_fused = fused(x.to(self.device_name))
            out_chain = bn(lin(x.to(self.device_name)))
        if self.device_name == 'npu':
            torch.npu.synchronize()
        torch.testing.assert_close(out_fused.cpu(), out_chain.cpu(),
                                   msg='fused output must equal the linear->bn chain output')

    def test_fuse_linear_bn_eval_fused_weight_formula(self):
        """Fused weight/bias equal the fuse_linear_bn_weights formula."""
        lin, bn, _, _ = self._seeded_pair()
        fused = torch.nn.utils.fuse_linear_bn_eval(lin, bn)
        self.assertEqual(tuple(fused.weight.shape), tuple(lin.weight.shape),
                         'fusing must preserve the linear weight shape')
        scale = bn.weight.detach().cpu() * torch.rsqrt(
            bn.running_var.detach().cpu() + bn.eps)
        exp_weight = lin.weight.detach().cpu() * scale.reshape(-1, 1)
        exp_bias = (lin.bias.detach().cpu() - bn.running_mean.detach().cpu()) * scale \
            + bn.bias.detach().cpu()
        torch.testing.assert_close(fused.weight.detach().cpu(), exp_weight,
                                   msg='fused weight must equal weight * bn_scale')
        torch.testing.assert_close(fused.bias.detach().cpu(), exp_bias,
                                   msg='fused bias must equal (bias - rm) * scale + beta')

    def test_fuse_linear_bn_eval_input_modules_unchanged(self):
        """Inputs are deep-copied: the original modules stay untouched."""
        lin, bn, _, _ = self._seeded_pair()
        weight_before = lin.weight.detach().clone()
        bias_before = lin.bias.detach().clone()
        rm_before = bn.running_mean.detach().clone()
        fused = torch.nn.utils.fuse_linear_bn_eval(lin, bn)
        self.assertIsNot(fused, lin, 'fusion must return a new module, not the input')
        torch.testing.assert_close(lin.weight.detach().cpu(), weight_before.cpu(),
                                   msg='input linear weight must not be modified')
        torch.testing.assert_close(lin.bias.detach().cpu(), bias_before.cpu(),
                                   msg='input linear bias must not be modified')
        torch.testing.assert_close(bn.running_mean.detach().cpu(), rm_before.cpu(),
                                   msg='input bn running stats must not be modified')

    def test_fuse_linear_bn_eval_no_bias_linear(self):
        """A bias-free Linear is fused with zeros standing in for the bias."""
        lin, bn, _, _ = self._seeded_pair(bias=False)
        self.assertIsNone(lin.bias, 'precondition: bias-free linear')
        fused = torch.nn.utils.fuse_linear_bn_eval(lin, bn)
        self.assertIsNotNone(fused.bias,
                             'fused bias must be materialized from bn parameters')
        scale = bn.weight.detach().cpu() * torch.rsqrt(
            bn.running_var.detach().cpu() + bn.eps)
        exp_bias = (torch.zeros_like(scale) - bn.running_mean.detach().cpu()) * scale \
            + bn.bias.detach().cpu()
        torch.testing.assert_close(fused.bias.detach().cpu(), exp_bias,
                                   msg='bias-free fusing must use zeros for the linear bias')

    def test_fuse_linear_bn_eval_num_features_one_broadcast(self):
        """bn.num_features == 1 broadcasts the affine over all output features."""
        torch.manual_seed(42)
        lin = torch.nn.Linear(4, 8).eval().to(self.device_name)
        torch.manual_seed(5)
        bn = torch.nn.BatchNorm1d(1).eval().to(self.device_name)
        with torch.no_grad():
            bn.running_mean.copy_(torch.randn(1))
            bn.running_var.copy_(torch.rand(1) + 0.2)
        fused = torch.nn.utils.fuse_linear_bn_eval(lin, bn)
        self.assertEqual(tuple(fused.weight.shape), (8, 4),
                         'broadcast fusing must keep the linear weight shape')
        x = torch.randn(3, 4).to(self.device_name)
        with torch.no_grad():
            ref = (lin(x) - bn.running_mean) * torch.rsqrt(bn.running_var + bn.eps) \
                * bn.weight + bn.bias
            out = fused(x)
        if self.device_name == 'npu':
            torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), ref.cpu(),
                                   msg='broadcast path must apply the single-channel '
                                       'affine to every output feature')

    # ------------------------------------------------------------------
    # Exception paths
    # ------------------------------------------------------------------

    def test_fuse_linear_bn_eval_training_mode_raises(self):
        """Training-mode linear or bn must raise AssertionError."""
        torch.manual_seed(42)
        lin = torch.nn.Linear(4, 8)
        bn = torch.nn.BatchNorm1d(8)
        with self.assertRaisesRegex(AssertionError, 'Fusion only for eval'):
            torch.nn.utils.fuse_linear_bn_eval(lin.train(), bn.eval())
        with self.assertRaisesRegex(AssertionError, 'Fusion only for eval'):
            torch.nn.utils.fuse_linear_bn_eval(lin.eval(), bn.train())

    def test_fuse_linear_bn_eval_num_features_mismatch_raises(self):
        """Mismatched out_features/num_features (not 1) must raise AssertionError."""
        torch.manual_seed(42)
        lin = torch.nn.Linear(4, 8).eval().to(self.device_name)
        bn = torch.nn.BatchNorm1d(16).eval().to(self.device_name)
        with self.assertRaisesRegex(AssertionError, 'linear.out_features'):
            torch.nn.utils.fuse_linear_bn_eval(lin, bn)

    # ------------------------------------------------------------------
    # CPU golden comparison
    # ------------------------------------------------------------------

    def test_fuse_linear_bn_eval_cpu_golden(self):
        """Same-seeded device fusion matches the CPU fused parameters."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        _, _, cpu_lin, cpu_bn = self._seeded_pair(device='cpu')
        lin, bn, _, _ = self._seeded_pair(device=self.device_name)
        cpu_fused = torch.nn.utils.fuse_linear_bn_eval(cpu_lin, cpu_bn)
        dev_fused = torch.nn.utils.fuse_linear_bn_eval(lin, bn)
        torch.testing.assert_close(dev_fused.weight.detach().cpu(),
                                   cpu_fused.weight.detach(),
                                   msg='device fused weight must match the CPU golden')
        torch.testing.assert_close(dev_fused.bias.detach().cpu(),
                                   cpu_fused.bias.detach(),
                                   msg='device fused bias must match the CPU golden')


if __name__ == "__main__":
    run_tests()
