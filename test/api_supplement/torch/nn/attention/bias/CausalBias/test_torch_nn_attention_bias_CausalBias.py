# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.attention.bias.CausalBias 接口功能正确性
API 名称：torch.nn.attention.bias.CausalBias
API 签名：CausalBias(variant: CausalVariant, seq_len_q: int, seq_len_kv: int)

覆盖维度表：
| 覆盖维度         | 说明                                               | 覆盖情况                       |
|------------------|----------------------------------------------------|--------------------------------|
| 枚举选项         | CausalVariant.UPPER_LEFT / LOWER_RIGHT 全覆盖      | 已覆盖                         |
| 参数类型         | variant 非枚举值（str）触发 TypeError              | 已覆盖                         |
| 传参与不传参     | 工厂函数 causal_upper_left / causal_lower_right    | 已覆盖                         |
| 等价类/边界值    | 方阵等价性；q > kv 警告边界                        | 已覆盖                         |
| 正常传参场景     | _materialize 值、dtype、device 契约                | 已覆盖                         |
| 异常传参场景     | 非法 variant 报 TypeError                          | 已覆盖                         |
| 副作用验证       | LOWER_RIGHT 且 q>kv 发 UserWarning                 | 已覆盖                         |
| 混合设备类型     | 不适用：单 device 参数构造，无多 Tensor 混合输入   | 不适用                         |

未覆盖项及原因：
- CausalBias 与 SDPA 的端到端注意力计算：本 API 看护对象为 bias 构造与
  _materialize 语义，SDPA 数值链路由 scaled_dot_product_attention 自身用例看护。

注意：_materialize 输出为 bool 张量，CPU 结果为 Golden，设备侧输出移至 CPU
     后用 torch.equal 对照；警告校验使用 warnings.catch_warnings（torch_npu
     TestCase 的 assertWarns 签名不同，不使用）。
"""
import os
import unittest
import warnings

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Resolve device only when tests actually run; keep default CI collection cheap.
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


from torch.nn.attention.bias import CausalBias  # noqa: E402
from torch.nn.attention.bias import CausalVariant  # noqa: E402
from torch.nn.attention.bias import causal_lower_right  # noqa: E402
from torch.nn.attention.bias import causal_upper_left  # noqa: E402


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestCausalBias(TestCase):
    """Test torch.nn.attention.bias.CausalBias semantics (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_causal_bias_init_records_variant_and_lengths(self):
        # Both enum members must be stored verbatim along with the two lengths.
        for variant in (CausalVariant.UPPER_LEFT, CausalVariant.LOWER_RIGHT):
            bias = CausalBias(variant, 3, 5)
            self.assertIsInstance(bias, CausalBias)
            self.assertIs(bias.variant, variant)
            self.assertEqual(bias.seq_len_q, 3)
            self.assertEqual(bias.seq_len_kv, 5)

    def test_causal_bias_is_tensor_subclass(self):
        # CausalBias subclasses torch.Tensor; isinstance checks must hold.
        bias = causal_upper_left(2, 2)
        self.assertIsInstance(bias, torch.Tensor)
        self.assertIsInstance(bias, CausalBias)

    def test_causal_bias_upper_left_materialize_values(self):
        # UPPER_LEFT is tril(ones(q, kv)): known 3x4 golden.
        bias = causal_upper_left(3, 4)
        materialized = bias._materialize(torch.device("cpu"))
        expected = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]],
                                dtype=torch.bool)
        self.assertEqual(materialized.dtype, torch.bool)
        self.assertTrue(bool(torch.equal(materialized, expected)))

    def test_causal_bias_lower_right_materialize_values(self):
        # LOWER_RIGHT uses diagonal offset kv-q: known 3x4 golden.
        bias = causal_lower_right(3, 4)
        materialized = bias._materialize(torch.device("cpu"))
        expected = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
                                dtype=torch.bool)
        self.assertTrue(bool(torch.equal(materialized, expected)))

    def test_causal_bias_square_variants_equivalent(self):
        # For q == kv both variants materialize to the same triangular mask.
        upper = causal_upper_left(4, 4)._materialize(torch.device("cpu"))
        lower = causal_lower_right(4, 4)._materialize(torch.device("cpu"))
        self.assertTrue(bool(torch.equal(upper, lower)))
        self.assertTrue(bool(torch.equal(upper, torch.tril(torch.ones(4, 4, dtype=torch.bool)))))

    def test_causal_bias_materialize_default_device_is_cpu(self):
        # _materialize() without a device argument falls back to CPU.
        bias = causal_upper_left(2, 3)
        materialized = bias._materialize()
        self.assertEqual(materialized.device.type, "cpu")

    def test_causal_bias_materialize_on_device_matches_cpu_golden(self):
        # Device materialization must equal the CPU golden mask element-wise.
        upper = causal_upper_left(3, 4)
        lower = causal_lower_right(3, 4)
        upper_dev = upper._materialize(self.device)
        lower_dev = lower._materialize(self.device)
        self.assertEqual(upper_dev.device.type, self.device_name)
        torch.testing.assert_close(upper_dev.cpu(), upper._materialize(torch.device("cpu")))
        torch.testing.assert_close(lower_dev.cpu(), lower._materialize(torch.device("cpu")))

    def test_causal_bias_lower_right_longer_q_warns(self):
        # Documented warning: LOWER_RIGHT with seq_len_q > seq_len_kv produces NaNs.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            causal_lower_right(5, 3)
        self.assertEqual(len(caught), 1)
        self.assertIn("NaNs", str(caught[0].message))
        self.assertIn("seq_len_q > seq_len_kv", str(caught[0].message))

    def test_causal_bias_no_warning_for_valid_shapes(self):
        # No warning may fire for UPPER_LEFT q>kv or LOWER_RIGHT q<=kv.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            causal_upper_left(5, 3)
            causal_lower_right(3, 5)
            causal_lower_right(4, 4)
        self.assertEqual(len(caught), 0)

    def test_causal_bias_invalid_variant_raises_type_error(self):
        # torch.Tensor.__new__ rejects a str variant before __init__ validation.
        with self.assertRaises(TypeError):
            CausalBias("UPPER_LEFT", 3, 3)

    def test_causal_bias_materialize_repeated_calls_deterministic(self):
        # Repeated materialization on device must be bit-identical (no hidden state).
        bias = causal_lower_right(3, 4)
        first = bias._materialize(self.device)
        second = bias._materialize(self.device)
        self.assertTrue(bool(torch.equal(first.cpu(), second.cpu())))


if __name__ == "__main__":
    run_tests()
