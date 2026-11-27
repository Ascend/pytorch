# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.utils.determine_qparams 接口功能正确性
API 名称：torch.ao.quantization.utils.determine_qparams
API 签名：determine_qparams(min_val, max_val, quant_min, quant_max, dtype, eps,
          has_customized_qrange, qscheme=torch.per_tensor_affine)

覆盖维度表：
| 覆盖维度      | 说明                                      | 覆盖情况                                             |
|---------------|-------------------------------------------|------------------------------------------------------|
| 空/非空       | 有效 min/max / 空张量（回退默认值）       | 已覆盖：test_per_tensor_affine_values、              |
|               |                                           | test_invalid_min_max_returns_default                 |
| 枚举选项      | qscheme：per_tensor_affine/symmetric、    | 已覆盖：test_per_tensor_affine_values、              |
|               | per_channel_symmetric、                   | test_symmetric_qscheme、test_per_channel_symmetric、 |
|               | per_channel_affine_float                  |                                                      |
|               |                                           | test_per_channel_affine_float_qparams                |
| 参数类型      | min_val/max_val 为 Tensor，eps 为 Tensor  | 已覆盖：test_per_tensor_affine_values                |
| 传参与不传参  | qscheme 省略默认 / 显式传入               | 已覆盖：test_default_qscheme_matches_explicit        |
| 等价类/边界值 | min==max==0（scale 被 eps 截断）、标量/   | 已覆盖：test_scale_clamped_to_eps、                  |
|               | 向量                                      | test_per_channel_symmetric                           |
| 正常传参场景  | 返回 (scale, zero_point) 且 dtype 为      | 已覆盖：test_per_tensor_affine_returns_dtypes        |
|               | double/int64                              |                                                      |
| 异常传参场景  | 空 min/max 返回默认值而非报错             | 已覆盖：test_invalid_min_max_returns_default         |
| 混合设备类型  | 多 Tensor 异构设备输入                    | 不适用，qparams 计算在 CPU 上进行，无设备参与        |

未覆盖项及原因：
- 混合设备类型：该工具为 CPU 数值变换，输入不要求设备执行，异构设备输入无意义，故不覆盖。
- 图模式：本用例针对 eager 数值工具，未覆盖 torch.compile / NPUGraph 图模式路径。

注：浮点 scale 使用 torch.testing.assert_close 精确对照；zero_point 为整型，使用精确断言。
"""
import os
import unittest
import warnings

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe the device only when the supplement suite actually runs. On the default
# CI collection path the class-level skipUnless fires first, so no torch_npu
# import or device initialization happens here.
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
        DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

from torch.ao.quantization.utils import determine_qparams

_EPS = torch.tensor(1e-5)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDetermineQparamsUt(TestCase):
    """Functional tests for torch.ao.quantization.utils.determine_qparams."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_per_tensor_affine_returns_dtypes(self):
        scale, zero_point = determine_qparams(
            torch.tensor(0.0), torch.tensor(6.0), 0, 255, torch.quint8, _EPS, False
        )
        self.assertEqual(scale.dtype, torch.double)
        self.assertEqual(zero_point.dtype, torch.int64)
        self.assertEqual(tuple(scale.shape), (1,))

    def test_per_tensor_affine_values(self):
        scale, zero_point = determine_qparams(
            torch.tensor(0.0), torch.tensor(6.0), 0, 255, torch.quint8, _EPS, False
        )
        torch.testing.assert_close(scale, torch.tensor([6.0 / 255.0], dtype=torch.double))
        self.assertEqual(int(zero_point), 0)

    def test_qint8_affine_zero_point(self):
        scale, zero_point = determine_qparams(
            torch.tensor(0.0), torch.tensor(6.0), -128, 127, torch.qint8, _EPS, False
        )
        torch.testing.assert_close(scale, torch.tensor([6.0 / 255.0], dtype=torch.double))
        self.assertEqual(int(zero_point), -128)

    def test_scale_clamped_to_eps(self):
        # Degenerate range: scale collapses to eps instead of 0.
        scale, zero_point = determine_qparams(
            torch.tensor(0.0), torch.tensor(0.0), 0, 255, torch.quint8, _EPS, False
        )
        torch.testing.assert_close(scale, torch.tensor([1e-5], dtype=torch.double))
        self.assertEqual(int(zero_point), 0)

    def test_symmetric_qscheme(self):
        scale, zero_point = determine_qparams(
            torch.tensor(-3.0), torch.tensor(6.0), 0, 255, torch.quint8, _EPS, False,
            torch.per_tensor_symmetric,
        )
        torch.testing.assert_close(scale, torch.tensor([6.0 / 127.5], dtype=torch.double))
        self.assertEqual(int(zero_point), 128)

    def test_per_channel_symmetric(self):
        scale, zero_point = determine_qparams(
            torch.tensor([-1.0, -2.0]), torch.tensor([3.0, 4.0]), 0, 255, torch.quint8,
            _EPS, False, torch.per_channel_symmetric,
        )
        torch.testing.assert_close(
            scale, torch.tensor([3.0 / 127.5, 4.0 / 127.5], dtype=torch.double)
        )
        self.assertEqual(tuple(zero_point.shape), (2,))

    def test_per_channel_affine_float_qparams(self):
        scale, zero_point = determine_qparams(
            torch.tensor([0.0, -1.0]), torch.tensor([6.0, 5.0]), 0, 255, torch.quint8,
            _EPS, False, torch.per_channel_affine_float_qparams,
        )
        torch.testing.assert_close(
            scale, torch.tensor([6.0 / 255.0, 6.0 / 255.0], dtype=torch.double)
        )
        self.assertEqual(zero_point.tolist(), [0, 42])

    def test_default_qscheme_matches_explicit(self):
        default = determine_qparams(torch.tensor(0.0), torch.tensor(6.0), 0, 255, torch.quint8, _EPS, False)
        explicit = determine_qparams(
            torch.tensor(0.0), torch.tensor(6.0), 0, 255, torch.quint8, _EPS, False,
            torch.per_tensor_affine,
        )
        torch.testing.assert_close(default[0], explicit[0])
        self.assertEqual(default[1].tolist(), explicit[1].tolist())

    def test_invalid_min_max_returns_default(self):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            scale, zero_point = determine_qparams(
                torch.tensor([]), torch.tensor([]), 0, 255, torch.quint8, _EPS, False
            )
        self.assertEqual(len(records), 1)
        self.assertTrue(issubclass(records[0].category, UserWarning))
        # The fallback returns a float32 scale of 1.0 and zero_point of 0.
        torch.testing.assert_close(scale, torch.tensor([1.0]))
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(int(zero_point), 0)


if __name__ == "__main__":
    run_tests()
