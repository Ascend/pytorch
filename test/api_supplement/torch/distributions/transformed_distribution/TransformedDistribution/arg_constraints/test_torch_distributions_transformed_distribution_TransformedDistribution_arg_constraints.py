# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.transformed_distribution.
TransformedDistribution.arg_constraints 接口功能正确性
API 名称：torch.distributions.transformed_distribution.TransformedDistribution.arg_constraints
API 签名：arg_constraints: dict[str, constraints.Constraint] = {}（类属性）

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 基类为空字典，子类为非空键集           | 已覆盖(empty_for_base_class、        |
|              |                                        | subclass_keys_follow_source）         |
| 枚举选项     | 本接口无枚举入参                       | 不适用                                |
| 参数类型     | dict[str, Constraint] 键为字符串       | 已覆盖（class_level_attribute、       |
|              |                                        | subclass_instance_unchanged）         |
| 传参与不传参 | 无入参类属性；实例化不改变键集         | 已覆盖(class_level_attribute)      |
| 等价类/边界  | 单 transform 与多 transform 构造       | 已覆盖（empty_for_base_class）        |
| 正常传参     | 键集合与源码定义逐键对照               | 已覆盖（subclass_keys_follow_source、 |
|              |                                        | subclass_instance_unchanged）         |
| 异常传参     | 无异常路径(类属性读取不触发校验)     | 不适用                                |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：运行时向实例写入自定义 arg_constraints 后校验行为。源码
  未承诺实例级覆盖的语义，属未定义行为，不做伪断言。
"""
import os
import unittest

import torch

from torch.distributions.pareto import Pareto
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.uniform import Uniform

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化，避免
# 不必要的探测开销；被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
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

# ORDER CONSTRAINT: import torch_npu.testing.testcase BEFORE
# torch_npu.contrib.transfer_to_npu (patching first breaks the import).
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    # No torch_npu (e.g. a plain CUDA/CPU box). Fall back to the stdlib
    # TestCase. This file does not compare floating point tensors, so the
    # fallback is safe here.
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

# transfer_to_npu rebinds torch.cuda.* and torch.Event to the NPU
# implementations, so the test body uses generic entry points unmodified
# on NPU and CUDA. Keep AFTER the testcase import above.
if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTransformedDistributionArgConstraints(TestCase):
    """Test cases for TransformedDistribution.arg_constraints."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_arg_constraints_empty_for_base_class(self):
        """A plain TransformedDistribution has an empty constraint dict."""
        base = Uniform(torch.tensor(0.0), torch.tensor(1.0))
        single = TransformedDistribution(base, SigmoidTransform().inv)
        self.assertEqual(single.arg_constraints, {})
        double = TransformedDistribution(
            base, [SigmoidTransform().inv, SigmoidTransform()])
        self.assertEqual(double.arg_constraints, {})

    def test_arg_constraints_is_class_level_attribute(self):
        """Instances share the same class-level empty dict object."""
        self.assertEqual(TransformedDistribution.arg_constraints, {})
        base = Uniform(torch.tensor(0.0), torch.tensor(1.0))
        d = TransformedDistribution(base, SigmoidTransform().inv)
        self.assertIs(d.arg_constraints, TransformedDistribution.arg_constraints)

    def test_subclass_keys_follow_source(self):
        """Subclasses override arg_constraints themselves."""
        self.assertEqual(set(Pareto.arg_constraints), {"alpha", "scale"})
        self.assertEqual(
            set(RelaxedBernoulli.arg_constraints), {"probs", "logits"})

    def test_subclass_instance_arg_constraints_unchanged(self):
        """Base-dist/transform constraints are never merged in."""
        p = Pareto(
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([2.0], dtype=torch.float64))
        # The Exponential base dist contributes "rate", which is not merged.
        self.assertEqual(set(p.arg_constraints), {"alpha", "scale"})
        self.assertIs(p.arg_constraints, Pareto.arg_constraints)

    def test_instance_construction_does_not_mutate_class_dict(self):
        """Constructing instances leaves the class-level dict empty."""
        base = Uniform(torch.tensor(0.0), torch.tensor(1.0))
        TransformedDistribution(base, [SigmoidTransform().inv])
        self.assertEqual(TransformedDistribution.arg_constraints, {})


if __name__ == "__main__":
    run_tests()
