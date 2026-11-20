# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.LinearCrossEntropyOptions.acc_dtype 接口功能正确性
API 名称：torch.nn.LinearCrossEntropyOptions.acc_dtype
API 签名：acc_dtype: torch.dtype | None = None（dataclass 字段，构造参数 acc_dtype=...）

说明：acc_dtype 是 LinearCrossEntropyOptions 的累加精度字段，None 表示调用时自动解析
（fp16/bf16 输入在支持混合精度的硬件上解析为 torch.float32，否则回落输入 dtype）；
仅接受 None/fp16/bf16/fp32/fp64，其余值在 dataclass __post_init__ 校验即抛 ValueError。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 默认 None 与显式 dtype | 已覆盖（_default_none/_valid_values） |
| 枚举选项 | None/fp16/bf16/fp32/fp64 全部合法值 | 已覆盖（_valid_values） |
| 参数类型 | torch.dtype 单例保持（assertIs） | 已覆盖（_valid_values） |
| 传参与不传参 | 省略（默认 None）与显式传参 | 已覆盖（_default_none/_valid_values） |
| 等价类/边界值 | fp64 边界合法值、None 自动解析回落输入 dtype | 已覆盖（_valid_values/_adjust_resolution） |
| 正常传参场景 | _adjust 调用点解析：cpu+fp16→fp32、无设备→输入 dtype | 已覆盖（_adjust_resolution） |
| 异常传参场景 | 非法 dtype（int64/complex64）构造即抛 ValueError | 已覆盖（_invalid_raises） |
| 混合设备类型 | 解析仅依赖 device.type 的纯 Python 判定，无设备执行路径 | 未覆盖（解析逻辑按 device 参数类型判定，见下） |

未覆盖项及原因：
- 混合设备类型：acc_dtype 的自动解析按 device.type 与 dtype 的纯 Python 规则判定，
  不产生设备执行路径；真实分块计算中的混合精度效果由 F.linear_cross_entropy 功能
  用例在设备上验证（chunked fp16 + acc_dtype=float32 场景）。
- 解析规则以 PyTorch 2.15.0 的 _adjust/_resolve_auto_acc_dtype 实现为准（内部 API，
  上游测试亦按该调用点使用），跨版本行为可能变化。

注意：纯 Python 字段契约按取值、校验与解析结果断言；无浮点 Tensor 输出，不做
     CPU Golden 数值对照。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Pure-container API: device probing stays opportunistic and never raises.
DEVICE_NAME = "cpu"
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLinearCrossEntropyOptionsAccDtype(TestCase):
    """Test cases for the acc_dtype field of LinearCrossEntropyOptions."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_acc_dtype_default_none(self):
        """Omitted acc_dtype resolves to the documented None sentinel."""
        self.assertIsNone(torch.nn.LinearCrossEntropyOptions().acc_dtype)

    def test_acc_dtype_valid_values(self):
        """All documented dtypes are accepted and stored as dtype singletons."""
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            options = torch.nn.LinearCrossEntropyOptions(acc_dtype=dtype)
            self.assertIs(options.acc_dtype, dtype)

    def test_acc_dtype_invalid_raises(self):
        """Non-float accumulators are rejected with ValueError at construction."""
        for dtype in (torch.int64, torch.complex64):
            with self.assertRaisesRegex(ValueError, "acc_dtype"):
                torch.nn.LinearCrossEntropyOptions(acc_dtype=dtype)

    def test_acc_dtype_adjust_resolution(self):
        """_adjust resolves None per (dtype, device) per the documented rule."""
        # CPU low-precision input resolves None to fp32 accumulation.
        resolved = torch.nn.LinearCrossEntropyOptions()._adjust(
            num_batches=8,
            in_features=4,
            num_classes=10,
            dtype=torch.float16,
            device=torch.device("cpu"),
        )
        self.assertIs(resolved.acc_dtype, torch.float32)
        # Without a device hint the input dtype is kept.
        resolved = torch.nn.LinearCrossEntropyOptions()._adjust(
            num_batches=8,
            in_features=4,
            num_classes=10,
            dtype=torch.float16,
            device=None,
        )
        self.assertIs(resolved.acc_dtype, torch.float16)
        # Non-low-precision input keeps its own dtype.
        resolved = torch.nn.LinearCrossEntropyOptions()._adjust(
            num_batches=8,
            in_features=4,
            num_classes=10,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertIs(resolved.acc_dtype, torch.float32)
        # An explicit value always wins over auto resolution.
        resolved = torch.nn.LinearCrossEntropyOptions(acc_dtype=torch.float32)._adjust(
            num_batches=8,
            in_features=4,
            num_classes=10,
            dtype=torch.float16,
            device=None,
        )
        self.assertIs(resolved.acc_dtype, torch.float32)


if __name__ == "__main__":
    run_tests()
