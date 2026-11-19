# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.modules.transformer.Transformer.generate_square_subsequent_mask 接口功能正确性
API 名称：torch.nn.modules.transformer.Transformer.generate_square_subsequent_mask
API 签名：generate_square_subsequent_mask(sz: int, device=None, dtype=None) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                | 覆盖情况                       |
|------------------|-----------------------------------------------------|--------------------------------|
| 等价类/边界值    | sz=0 空掩码、sz=1 单元素、常规 sz                   | 已覆盖                         |
| 参数类型         | dtype float32/float16；非浮点 dtype 报 RuntimeError  | 已覆盖                         |
| 传参与不传参     | device/dtype 均可省略，默认 float32                 | 已覆盖                         |
| 正常传参场景     | -inf 上三角、0.0 其余位置的精确数值                 | 已覆盖                         |
| 异常传参场景     | dtype=int64 报 RuntimeError                         | 已覆盖                         |
| 混合设备类型     | 不适用：单 device 参数构造，无多 Tensor 入参        | 不适用                         |
| 静态方法契约     | 类上与实例上均可调用                                | 已覆盖                         |
| 确定性           | 同参数重复调用结果一致                              | 已覆盖                         |

未覆盖项及原因：
- dtype=None 时跟随全局默认 dtype 的路径：构造 float 默认掩码已覆盖主契约，
  全局 set_default_dtype 状态切换影响面超出本 API 看护范围。

注意：输出为 float 掩码（-inf/0.0 精确值），CPU 侧为 Golden，设备侧输出移至
     CPU 后用 torch.equal 对照；不做浮点容差比较。
"""
import os
import unittest

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


from torch.nn.modules.transformer import Transformer  # noqa: E402


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTransformerGenerateSquareSubsequentMask(TestCase):
    """Test Transformer.generate_square_subsequent_mask contracts (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_mask_values_and_dtype_default(self):
        # Default dtype float32; -inf strictly above the diagonal, 0.0 elsewhere.
        mask = Transformer.generate_square_subsequent_mask(4, device=self.device_name)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask.device.type, self.device_name)
        expected = torch.triu(torch.full((4, 4), float("-inf")), diagonal=1)
        self.assertTrue(bool(torch.equal(mask.cpu(), expected)))
        # Exact spot checks: diagonal 0.0, first row masked after position 0
        # (-inf compared directly; torch_npu assertEqual rejects non-finite).
        self.assertEqual(mask.cpu()[0, 0].item(), 0.0)
        self.assertTrue(mask.cpu()[0, 1].item() == float("-inf"))
        self.assertEqual(mask.cpu()[3, 0].item(), 0.0)
        self.assertEqual(mask.cpu()[3, 3].item(), 0.0)

    def test_mask_matches_cpu_golden(self):
        # Device-built mask must equal the CPU-built golden element-wise.
        mask_dev = Transformer.generate_square_subsequent_mask(9, device=self.device_name)
        mask_cpu = Transformer.generate_square_subsequent_mask(9, device="cpu")
        self.assertEqual(mask_dev.device.type, self.device_name)
        self.assertTrue(bool(torch.equal(mask_dev.cpu(), mask_cpu)))

    def test_mask_explicit_dtype_float16(self):
        # An explicit floating dtype is honored and keeps the -inf contract.
        mask = Transformer.generate_square_subsequent_mask(3, device=self.device_name,
                                                           dtype=torch.float16)
        self.assertEqual(mask.dtype, torch.float16)
        self.assertTrue(mask.cpu()[0, 2].item() == float("-inf"))
        self.assertEqual(mask.cpu()[2, 2].item(), 0.0)

    def test_mask_integer_dtype_raises(self):
        # -inf is not representable in integer dtypes; construction must fail.
        with self.assertRaises(RuntimeError):
            Transformer.generate_square_subsequent_mask(3, dtype=torch.int64)

    def test_mask_boundary_sizes(self):
        # sz=0 yields an empty (0, 0) mask; sz=1 is a single unmasked position.
        empty = Transformer.generate_square_subsequent_mask(0, device=self.device_name)
        self.assertEqual(tuple(empty.shape), (0, 0))
        single = Transformer.generate_square_subsequent_mask(1, device=self.device_name)
        self.assertEqual(tuple(single.shape), (1, 1))
        self.assertEqual(single.cpu()[0, 0].item(), 0.0)

    def test_mask_static_callable_on_class_and_instance(self):
        # The API is a static method: callable on the class and on an instance.
        on_class = Transformer.generate_square_subsequent_mask(2, device=self.device_name)
        model = Transformer(d_model=8, nhead=2, num_encoder_layers=1,
                            num_decoder_layers=1, dim_feedforward=8, batch_first=True)
        on_instance = model.generate_square_subsequent_mask(2, device=self.device_name)
        self.assertTrue(bool(torch.equal(on_class.cpu(), on_instance.cpu())))

    def test_mask_repeated_calls_deterministic(self):
        # Two calls with identical arguments must be bit-identical.
        first = Transformer.generate_square_subsequent_mask(5, device=self.device_name)
        second = Transformer.generate_square_subsequent_mask(5, device=self.device_name)
        self.assertTrue(bool(torch.equal(first.cpu(), second.cpu())))


if __name__ == "__main__":
    run_tests()
