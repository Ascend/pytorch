# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.transforms.identity_transform 接口功能正确性
API 名称：torch.distributions.transforms.identity_transform
API 签名：identity_transform = ComposeTransform([])

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 空 parts 列表(恒等)与非空输入张量    | 已覆盖(is_empty_compose_transform、  |
|              |                                        | forward/inverse）                     |
| 枚举选项     | 本接口无枚举入参                       | 不适用                                |
| 参数类型     | 0 维/2 维/3 维、float32 输入张量       | 已覆盖（scalar_and_batched_inputs）   |
| 传参与不传参 | __call__ 仅需 x；inv 无入参构造        | 已覆盖（forward、inverse）            |
| 等价类/边界  | 标量输入与多维批量输入两端             | 已覆盖(scalar_and_batched_inputs)   |
| 正常传参     | 前向/逆向输出与输入一致、Jacobian 为 0 | 已覆盖(forward、inverse、jacobian)  |
| 异常传参     | 无异常路径(恒等变换无参数校验)       | 不适用                                |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：带 autograd 图输入的恒等变换。恒等路径直接返回输入对象，
  梯度语义等同输入本身，无独立可证伪行为，不做伪断言。
"""
import os
import unittest

import torch

from torch.distributions import constraints
from torch.distributions.transforms import ComposeTransform, identity_transform

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
class TestIdentityTransform(TestCase):
    """Test cases for transforms.identity_transform (identity mapping)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_identity_forward(self):
        """Calling the transform returns the input unchanged."""
        torch.manual_seed(0)
        x = torch.randn(3, 4)
        torch.testing.assert_close(identity_transform(x), x)

    def test_identity_inverse(self):
        """The inverse composition also returns the input unchanged."""
        torch.manual_seed(1)
        x = torch.randn(2, 5)
        torch.testing.assert_close(identity_transform.inv(x), x)

    def test_identity_log_abs_det_jacobian_is_zero(self):
        """The log |det jacobian| of the identity is zero everywhere."""
        torch.manual_seed(2)
        x = torch.randn(4, 3)
        y = identity_transform(x)
        torch.testing.assert_close(
            identity_transform.log_abs_det_jacobian(x, y), torch.zeros_like(x))

    def test_identity_domain_and_codomain_are_real(self):
        """With no parts, domain and codomain are the real constraint."""
        self.assertIs(identity_transform.domain, constraints.real)
        self.assertIs(identity_transform.codomain, constraints.real)

    def test_identity_is_empty_compose_transform(self):
        """identity_transform is a ComposeTransform with no parts."""
        self.assertIsInstance(identity_transform, ComposeTransform)
        self.assertEqual(identity_transform.parts, [])

    def test_identity_on_scalar_and_batched_inputs(self):
        """Scalar and higher-rank inputs pass through unchanged."""
        scalar = torch.tensor(3.5)
        torch.testing.assert_close(identity_transform(scalar), scalar)
        torch.manual_seed(3)
        batch = torch.randn(2, 3, 4)
        torch.testing.assert_close(identity_transform(batch), batch)


if __name__ == "__main__":
    run_tests()
