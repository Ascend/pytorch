# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.utils.prune.RandomUnstructured.apply 接口功能正确性
API 名称：torch.nn.utils.prune.RandomUnstructured.apply
API 签名：apply(cls, module, name, amount) -> BasePruningMethod
（torch 2.15.0 实测：RandomUnstructured 以该收窄签名覆写基类 BasePruningMethod.apply，
importance_scores 仅存在于基类签名；传入覆写入口按 TypeError 拒绝，用例同时以
inspect.signature 固定收窄契约）

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | amount 整数（精确个数）与浮点（精确比例）两种合法取值 | 已覆盖
（apply_random_unstructured_reparameterizes/apply_random_unstructured_float_amount） |
| 枚举选项 | amount 整数（精确个数）与浮点（精确比例）两种合法取值；importance_scores 属基类
入口参数，覆写签名不含该参数 | 已覆盖
（apply_random_unstructured_reparameterizes/apply_random_unstructured_float_amount/
apply_random_unstructured_importance_scores） |
| 参数类型 | name='weight' 作用于 nn.Linear 参数；amount=int/float | 已覆盖
（apply_random_unstructured_reparameterizes） |
| 传参与不传参 | amount int/float 传参路径；importance_scores 传入按 TypeError 拒绝 | 已覆盖
（apply_random_unstructured_importance_scores） |
| 等价类/边界值 | weight==weight_orig*weight_mask；重复 apply 组合为 PruningContainer 且保留已有置零 | 已覆盖
（apply_random_unstructured_reparameterizes/apply_random_unstructured_reapply_composes） |
| 正常传参场景 | weight_orig/weight_mask 重参数化、前向 pre-hook 注册、前向使用被掩码权重、返回
RandomUnstructured 实例并记录 _tensor_name | 已覆盖
（apply_random_unstructured_reparameterizes/apply_random_unstructured_forward_uses_masked_weights/
apply_random_unstructured_returns_method） |
| 异常传参场景 | importance_scores 传入覆写入口报 TypeError（覆写存在与签名收窄由 inspect.signature 断言固定）；
amount 越界校验归 __init__（见未覆盖项） | 已覆盖
（apply_random_unstructured_importance_scores） |
| 混合设备类型 | 同种子权重 + 相同 mask 下设备前向与 CPU Golden 前向一致；置零个数两侧精确一致 | 已覆盖
（apply_random_unstructured_cpu_golden） |

未覆盖项及原因：
- importance_scores 的掩码计算语义未覆盖：torch 2.15.0 实测 RandomUnstructured.apply 以
  收窄签名 (module, name, amount) 覆写基类 apply，不接受 importance_scores（传入在签名绑定
  阶段报 TypeError: unexpected keyword argument，见对应用例，覆写存在性由 inspect.signature
  断言固定）；该参数的语义归 BasePruningMethod.apply 基类入口与 l1_unstructured 等方法看护。
- amount 越界（>1 的 float 或超过元素数的 int）异常未覆盖：RandomUnstructured.apply
  将 amount 传入 __init__ 校验（_validate_pruning_amount_init），但 apply 是 classmethod，
  其异常由 __init__ 抛出且 amount=0 的 k=0 边界依赖 torch.kthvalue 行为，
  amount 校验家族归口于 RandomUnstructured.__init__ 而非 apply 入口本身。
- mask 具体置零位置由 torch.rand_like 随机产生，不做位置级断言（设备间随机源不同），
  仅断言置零个数与 mask/orig 乘积语义。

注意：apply 在 NPU 设备模块上真实执行（rand_like/topk/mul 均走设备算子）；
     浮点对照使用 torch.testing.assert_close。
"""
import inspect
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

from torch.nn.utils.prune import BasePruningMethod, PruningContainer, RandomUnstructured


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRandomUnstructuredApply(TestCase):
    """Test cases for RandomUnstructured.apply (NPU/CUDA portable).

    apply() reparameterizes module[name] into name_orig (parameter) plus
    name_mask (buffer), registers a forward pre-hook holding the pruning
    method, and leaves module[name] equal to orig * mask.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _seeded_linear(self, in_features=10, out_features=4, device=None):
        if device is None:
            device = self.device_name
        torch.manual_seed(42)
        cpu_lin = torch.nn.Linear(in_features, out_features)
        lin = torch.nn.Linear(in_features, out_features).to(device)
        lin.load_state_dict(cpu_lin.state_dict())
        return lin

    # ------------------------------------------------------------------
    # Reparameterization contract
    # ------------------------------------------------------------------

    def test_apply_random_unstructured_reparameterizes(self):
        """apply() installs weight_orig, weight_mask, hook and masked weight."""
        lin = self._seeded_linear()
        method = RandomUnstructured.apply(lin, 'weight', amount=3)
        self.assertIn('weight_orig', dict(lin.named_parameters()),
                      'the original weight must be kept as weight_orig')
        self.assertIn('weight_mask', dict(lin.named_buffers()),
                      'the pruning mask must be stored as weight_mask')
        self.assertEqual(int((lin.weight_mask == 0).sum()), 3,
                         'an integer amount must prune exactly that many units')
        torch.testing.assert_close(
            lin.weight.detach(), lin.weight_orig * lin.weight_mask.to(lin.weight_orig.dtype),
            msg='module weight must equal weight_orig * weight_mask')
        hooks = [h for h in lin._forward_pre_hooks.values()
                 if isinstance(h, BasePruningMethod)]
        self.assertEqual(len(hooks), 1, 'exactly one pruning pre-hook must be registered')
        self.assertIs(hooks[0], method,
                      'apply() must return the registered pruning method instance')
        self.assertTrue(lin.weight_orig.requires_grad,
                        'weight_orig must stay trainable')

    def test_apply_random_unstructured_float_amount(self):
        """A float amount prunes exactly the documented fraction of units."""
        lin = self._seeded_linear(in_features=10, out_features=4)  # 40 units
        RandomUnstructured.apply(lin, 'weight', amount=0.25)
        self.assertEqual(int((lin.weight_mask == 0).sum()), 10,
                         '0.25 of 40 units must prune exactly 10')

    def test_apply_random_unstructured_returns_method(self):
        """apply() returns the RandomUnstructured instance bound to the tensor."""
        lin = self._seeded_linear()
        method = RandomUnstructured.apply(lin, 'weight', amount=2)
        self.assertIsInstance(method, RandomUnstructured)
        self.assertEqual(method._tensor_name, 'weight',
                         'the method must remember which tensor it prunes')
        self.assertEqual(RandomUnstructured.PRUNING_TYPE, 'unstructured')

    def test_apply_random_unstructured_importance_scores(self):
        """importance_scores is base-only and the narrowed override rejects it."""
        lin = self._seeded_linear()
        scores = torch.rand_like(lin.weight)
        # Measured on torch 2.15.0: RandomUnstructured defines its own
        # apply(cls, module, name, amount) override that delegates to
        # BasePruningMethod.apply, so importance_scores stays a keyword-only
        # parameter of the base signature and never reaches the override's
        # signature binding. Pin the narrowing itself: if the override were
        # removed, the signature check fails before the TypeError check.
        self.assertEqual(
            list(inspect.signature(RandomUnstructured.apply).parameters),
            ['module', 'name', 'amount'],
            'RandomUnstructured.apply must narrow the base signature to '
            '(module, name, amount) so importance_scores is not accepted')
        with self.assertRaisesRegex(
                TypeError, r"unexpected keyword argument 'importance_scores'"):
            RandomUnstructured.apply(lin, 'weight', amount=3,
                                     importance_scores=scores)

    def test_apply_random_unstructured_reapply_composes(self):
        """A second apply() composes into a PruningContainer and keeps zeros."""
        lin = self._seeded_linear()
        RandomUnstructured.apply(lin, 'weight', amount=4)
        first_zeros = int((lin.weight_mask == 0).sum())
        RandomUnstructured.apply(lin, 'weight', amount=4)
        hooks = [h for h in lin._forward_pre_hooks.values()
                 if isinstance(h, BasePruningMethod)]
        self.assertEqual(len(hooks), 1,
                         'repeated pruning must merge into a single hook')
        self.assertIsInstance(hooks[0], PruningContainer,
                              'repeated pruning must compose a PruningContainer')
        self.assertGreaterEqual(int((lin.weight_mask == 0).sum()), first_zeros,
                                'the composed mask must keep the already pruned units')

    # ------------------------------------------------------------------
    # Forward semantics
    # ------------------------------------------------------------------

    def test_apply_random_unstructured_forward_uses_masked_weights(self):
        """The module forward uses the masked weight and the original bias."""
        lin = self._seeded_linear()
        RandomUnstructured.apply(lin, 'weight', amount=3)
        x = torch.randn(2, 10).to(self.device_name)
        with torch.no_grad():
            out = lin(x)
            expected = F.linear(x, lin.weight_orig * lin.weight_mask.to(lin.weight_orig.dtype),
                                lin.bias)
        if self.device_name == 'npu':
            torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), expected.cpu(),
                                   msg='forward must use weight_orig * weight_mask')

    # ------------------------------------------------------------------
    # CPU golden comparison
    # ------------------------------------------------------------------

    def test_apply_random_unstructured_cpu_golden(self):
        """With the same mask, device forward matches the CPU golden exactly."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        dev_lin = self._seeded_linear()
        RandomUnstructured.apply(dev_lin, 'weight', amount=3)
        cpu_lin = self._seeded_linear(device='cpu')
        RandomUnstructured.apply(cpu_lin, 'weight', amount=3)
        self.assertEqual(int((cpu_lin.weight_mask == 0).sum()), 3,
                         'CPU golden must prune the same exact count')
        # Share one mask so both sides compute the same linear function; the
        # CPU golden is computed directly with F.linear to stay independent of
        # the CPU module's pruning hook.
        shared_mask = dev_lin.weight_mask.detach().cpu()
        x = torch.randn(2, 10)
        with torch.no_grad():
            out_dev = dev_lin(x.to(self.device_name))
            expected_cpu = F.linear(x, cpu_lin.weight_orig.detach() * shared_mask,
                                    cpu_lin.bias.detach())
        torch.testing.assert_close(out_dev.cpu(), expected_cpu,
                                   msg='device forward must match the CPU golden for '
                                       'identical weights and mask')


if __name__ == "__main__":
    run_tests()
