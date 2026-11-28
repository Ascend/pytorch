# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.relaxed_categorical.
RelaxedOneHotCategorical.temperature 接口功能正确性
API 名称：torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.temperature
API 签名：temperature(self) -> Tensor（property）

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 标量温度与批次温度张量                 | 已覆盖（scalar_broadcast、matches）   |
| 枚举选项     | 本接口无枚举入参                       | 不适用                                |
| 参数类型     | float32 与 float64 温度张量            | 已覆盖（dtype_float64）               |
| 传参与不传参 | 无入参 property；probs/logits 二选一   | 已覆盖（matches_constructor、         |
|              | 构造形态                               | logits_path）                         |
| 等价类/边界  | 温度等于 1 的边界取值                  | 已覆盖（temperature_one_boundary）    |
| 正常传参     | 与构造参数逐元素对照及对象同一性       | 已覆盖(matches_constructor、       |
|              |                                        | is_base_dist_temperature）            |
| 异常传参     | 无异常路径(属性读取不触发校验)       | 不适用                                |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：temperature 含 NaN/非正值的构造输入。 Relaxation 温度的
  正值约束由 arg_constraints 的 simplex 之外路径承载，构造期不校验
  temperature，无法形成稳定异常断言。
"""
import os
import unittest

import torch

from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

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
class TestRelaxedOneHotCategoricalTemperature(TestCase):
    """Test cases for RelaxedOneHotCategorical.temperature."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_temperature_matches_constructor(self):
        """temperature returns exactly the tensor passed at construction."""
        temp = torch.tensor([2.2])
        m = RelaxedOneHotCategorical(
            temp, probs=torch.tensor([0.1, 0.2, 0.3, 0.4]))
        torch.testing.assert_close(m.temperature, temp)

    def test_temperature_is_base_dist_temperature(self):
        """The property is the base distribution's temperature object."""
        m = RelaxedOneHotCategorical(
            torch.tensor([2.2]), probs=torch.tensor([0.1, 0.2, 0.3, 0.4]))
        self.assertIs(m.temperature, m.base_dist.temperature)

    def test_temperature_logits_path(self):
        """The logits parameterization preserves the temperature."""
        temp = torch.tensor([1.5])
        m = RelaxedOneHotCategorical(
            temp, logits=torch.tensor([0.0, 1.0, -1.0]))
        torch.testing.assert_close(m.temperature, temp)

    def test_temperature_scalar_broadcast_with_batched_probs(self):
        """A scalar temperature is returned unchanged for batched probs."""
        temp = torch.tensor(2.2)
        m = RelaxedOneHotCategorical(
            temp, probs=torch.tensor([[0.2, 0.3, 0.5], [0.5, 0.25, 0.25]]))
        torch.testing.assert_close(m.temperature, temp)

    def test_temperature_one_boundary(self):
        """temperature == 1 is accepted and returned as-is."""
        temp = torch.tensor([1.0])
        m = RelaxedOneHotCategorical(
            temp, probs=torch.tensor([0.25, 0.75]))
        torch.testing.assert_close(m.temperature, temp)

    def test_temperature_dtype_float64(self):
        """float64 temperature keeps its dtype through the property."""
        temp = torch.tensor([2.2], dtype=torch.float64)
        m = RelaxedOneHotCategorical(
            temp, probs=torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64))
        self.assertEqual(m.temperature.dtype, torch.float64)
        torch.testing.assert_close(m.temperature, temp)


if __name__ == "__main__":
    run_tests()
