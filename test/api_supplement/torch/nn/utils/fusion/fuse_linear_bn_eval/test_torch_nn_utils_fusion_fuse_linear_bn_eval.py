# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.utils.fusion.fuse_linear_bn_eval 接口功能正确性
API 名称：torch.nn.utils.fusion.fuse_linear_bn_eval
API 签名：fuse_linear_bn_eval(linear: torch.nn.Linear,
         bn: torch.nn.modules.batchnorm._BatchNorm) -> torch.nn.Linear

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | linear 有 bias；BatchNorm1d/BatchNorm2d 两类 bn 输入 | 已覆盖（fusion_fuse_linear_bn_eval_forward_matches_
original_chain/fusion_fuse_linear_bn_eval_batchnorm2d_fusion） |
| 枚举选项 | bn.num_features == linear.out_features 合法路径；training/eval 模式区分 | 已覆盖
（fusion_fuse_linear_bn_eval_forward_matches_original_chain/
fusion_fuse_linear_bn_eval_training_mode_raises） |
| 参数类型 | torch.nn.Linear + torch.nn.modules.batchnorm._BatchNorm 子类 | 已覆盖
（fusion_fuse_linear_bn_eval_batchnorm2d_fusion） |
| 传参与不传参 | 固定两参签名；torch.nn.utils 再导出与 fusion 模块入口为同一函数对象 | 已覆盖
（fusion_fuse_linear_bn_eval_alias_identity） |
| 等价类/边界值 | 融合权重等于 scale 公式 Golden；返回新模块 | 已覆盖
（fusion_fuse_linear_bn_eval_fused_weight_formula） |
| 正常传参场景 | 融合后 Linear 前向等于原 linear→bn 链路 | 已覆盖
（fusion_fuse_linear_bn_eval_forward_matches_original_chain/
fusion_fuse_linear_bn_eval_cpu_golden） |
| 异常传参场景 | training 模式 AssertionError | 已覆盖（fusion_fuse_linear_bn_eval_training_mode_raises） |
| 混合设备类型 | 相同种子的设备端融合权重/偏置与 CPU Golden 一致 | 已覆盖（fusion_fuse_linear_bn_eval_cpu_golden） |

未覆盖项及原因：
- bn.running_mean 为 None 分支不可达（同 torch.nn.utils.fuse_linear_bn_eval 文件说明）：
  LazyBatchNorm 物化前 num_features 为 None，先命中 num_features 校验。
- out_features 与 num_features 不匹配分支由 torch.nn.utils.fuse_linear_bn_eval 文件覆盖，
  本文件聚焦 fusion 命名空间入口与 BatchNorm2d 形态。

注意：融合为纯权重运算，但被融合模块与输入张量均置于 NPU 设备上执行；
     浮点对照统一使用 torch.testing.assert_close（float32 默认容差）。
"""
import os
import unittest

import torch
import torch.nn.utils  # noqa: F401  # ensure the re-export entry torch.nn.utils.fuse_linear_bn_eval loads
import torch.nn.utils.fusion  # noqa: F401  # the module under test

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
class TestFusionFuseLinearBnEval(TestCase):
    """Test cases for torch.nn.utils.fusion.fuse_linear_bn_eval (NPU/CUDA portable).

    The fusion module entry is the canonical implementation re-exported by
    torch.nn.utils; both entries share one function object. Fused parameters
    follow: scale = bn_w * rsqrt(running_var + eps), fused_weight = weight *
    scale, fused_bias = (bias - running_mean) * scale + bn_b.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _seeded_pair(self, device=None, bn_factory=torch.nn.BatchNorm1d):
        """Same-seeded Linear + BatchNorm pair moved to ``device``."""
        if device is None:
            device = self.device_name
        torch.manual_seed(42)
        cpu_lin = torch.nn.Linear(4, 8).eval()
        cpu_bn = bn_factory(8).eval()
        with torch.no_grad():
            cpu_bn.weight.copy_(torch.randn(8))
            cpu_bn.bias.copy_(torch.randn(8))
            cpu_bn.running_mean.copy_(torch.randn(8))
            cpu_bn.running_var.copy_(torch.rand(8) + 0.2)
        lin = torch.nn.Linear(4, 8).eval().to(device)
        bn = bn_factory(8).eval().to(device)
        lin.load_state_dict(cpu_lin.state_dict())
        bn.load_state_dict(cpu_bn.state_dict())
        return lin, bn, cpu_lin, cpu_bn

    # ------------------------------------------------------------------
    # Entry-point identity: the re-export shares the fusion implementation
    # ------------------------------------------------------------------

    def test_fusion_fuse_linear_bn_eval_alias_identity(self):
        """torch.nn.utils re-exports the same function object as fusion."""
        self.assertIs(torch.nn.utils.fusion.fuse_linear_bn_eval,
                      torch.nn.utils.fuse_linear_bn_eval,
                      'torch.nn.utils.fuse_linear_bn_eval must re-export the '
                      'fusion implementation')

    # ------------------------------------------------------------------
    # Functional contract
    # ------------------------------------------------------------------

    def test_fusion_fuse_linear_bn_eval_forward_matches_original_chain(self):
        """Fused Linear forward equals linear->bn on the same device input."""
        lin, bn, _, _ = self._seeded_pair()
        x = torch.randn(3, 4)
        fused = torch.nn.utils.fusion.fuse_linear_bn_eval(lin, bn)
        self.assertIsNot(fused, lin, 'fusion must return a new module')
        with torch.no_grad():
            out_fused = fused(x.to(self.device_name))
            out_chain = bn(lin(x.to(self.device_name)))
        if self.device_name == 'npu':
            torch.npu.synchronize()
        torch.testing.assert_close(out_fused.cpu(), out_chain.cpu(),
                                   msg='fused output must equal the linear->bn chain output')

    def test_fusion_fuse_linear_bn_eval_fused_weight_formula(self):
        """Fused weight/bias equal the fuse_linear_bn_weights formula."""
        lin, bn, _, _ = self._seeded_pair()
        fused = torch.nn.utils.fusion.fuse_linear_bn_eval(lin, bn)
        scale = bn.weight.detach().cpu() * torch.rsqrt(
            bn.running_var.detach().cpu() + bn.eps)
        exp_weight = lin.weight.detach().cpu() * scale.reshape(-1, 1)
        exp_bias = (lin.bias.detach().cpu() - bn.running_mean.detach().cpu()) * scale \
            + bn.bias.detach().cpu()
        torch.testing.assert_close(fused.weight.detach().cpu(), exp_weight,
                                   msg='fused weight must equal weight * bn_scale')
        torch.testing.assert_close(fused.bias.detach().cpu(), exp_bias,
                                   msg='fused bias must equal (bias - rm) * scale + beta')

    def test_fusion_fuse_linear_bn_eval_batchnorm2d_fusion(self):
        """BatchNorm2d (a _BatchNorm subclass) is fused with the same contract."""
        lin, bn, _, _ = self._seeded_pair(bn_factory=torch.nn.BatchNorm2d)
        fused = torch.nn.utils.fusion.fuse_linear_bn_eval(lin, bn)
        self.assertEqual(tuple(fused.weight.shape), (8, 4),
                         'BatchNorm2d fusing must preserve the linear weight shape')
        scale = bn.weight.detach().cpu() * torch.rsqrt(
            bn.running_var.detach().cpu() + bn.eps)
        exp_weight = lin.weight.detach().cpu() * scale.reshape(-1, 1)
        torch.testing.assert_close(fused.weight.detach().cpu(), exp_weight,
                                   msg='BatchNorm2d fused weight must equal weight * bn_scale')

    # ------------------------------------------------------------------
    # Exception path
    # ------------------------------------------------------------------

    def test_fusion_fuse_linear_bn_eval_training_mode_raises(self):
        """Training-mode linear or bn must raise AssertionError."""
        lin, bn, _, _ = self._seeded_pair(device='cpu')
        with self.assertRaisesRegex(AssertionError, 'Fusion only for eval'):
            torch.nn.utils.fusion.fuse_linear_bn_eval(lin.train(), bn.eval())
        with self.assertRaisesRegex(AssertionError, 'Fusion only for eval'):
            torch.nn.utils.fusion.fuse_linear_bn_eval(lin.eval(), bn.train())

    # ------------------------------------------------------------------
    # CPU golden comparison
    # ------------------------------------------------------------------

    def test_fusion_fuse_linear_bn_eval_cpu_golden(self):
        """Same-seeded device fusion matches the CPU fused parameters."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        _, _, cpu_lin, cpu_bn = self._seeded_pair(device='cpu')
        lin, bn, _, _ = self._seeded_pair(device=self.device_name)
        cpu_fused = torch.nn.utils.fusion.fuse_linear_bn_eval(cpu_lin, cpu_bn)
        dev_fused = torch.nn.utils.fusion.fuse_linear_bn_eval(lin, bn)
        torch.testing.assert_close(dev_fused.weight.detach().cpu(),
                                   cpu_fused.weight.detach(),
                                   msg='device fused weight must match the CPU golden')
        torch.testing.assert_close(dev_fused.bias.detach().cpu(),
                                   cpu_fused.bias.detach(),
                                   msg='device fused bias must match the CPU golden')


if __name__ == "__main__":
    run_tests()
