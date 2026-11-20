# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.LinearCrossEntropyOptions.acc_policy 接口功能正确性
API 名称：torch.nn.LinearCrossEntropyOptions.acc_policy
API 签名：acc_policy: Literal['accurate', 'compact', 'auto'] = 'auto'（dataclass 字段，
构造参数 acc_policy=...）

说明：acc_policy 是 LinearCrossEntropyOptions 的精度/内存取舍字段，'auto' 表示调用时按
(device.type, dtype) 解析：仅 CPU + fp16/bf16 解析为 'accurate'，其余（含 device=None）解析
为 'compact'；非法值在 dataclass __post_init__ 校验即抛 ValueError。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 默认 'auto' 与显式取值 | 已覆盖（_default_auto/_valid_values） |
| 枚举选项 | 'accurate'/'compact'/'auto' 全部合法值 | 已覆盖（_valid_values） |
| 参数类型 | 字符串枚举，非字符串构造即抛 | 已覆盖（_valid_values/_invalid_raises） |
| 传参与不传参 | 省略（默认 'auto'）与显式传参 | 已覆盖（_default_auto/_valid_values） |
| 等价类/边界值 | device=None 强制 'compact' 的回落分支 | 已覆盖（_auto_resolution） |
| 正常传参场景 | _adjust 调用点解析：cpu+fp16→accurate、cpu+fp32→compact | 已覆盖（_auto_resolution） |
| 异常传参场景 | 非法策略字符串/非字符串构造即抛 ValueError | 已覆盖（_invalid_raises） |
| 混合设备类型 | 解析仅按 device.type 的纯 Python 判定，仅构造可解析 device 对象（meta/cuda 恒可解析，npu 需 torch_npu 注册） | 已覆盖（_auto_resolution_device_types） |

未覆盖项及原因：
- 各策略在真实分块计算中的精度/内存差异由 F.linear_cross_entropy 功能用例在设备上验证；
  本文件看护字段的取值与解析契约。
- 解析规则以 PyTorch 2.15.0 的 _adjust/_auto_acc_policy 实现为准（内部 API，上游测试亦按
  该调用点使用），跨版本行为可能变化。

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
class TestLinearCrossEntropyOptionsAccPolicy(TestCase):
    """Test cases for the acc_policy field of LinearCrossEntropyOptions."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_acc_policy_default_auto(self):
        """Omitted acc_policy resolves to the documented 'auto' sentinel."""
        self.assertEqual(torch.nn.LinearCrossEntropyOptions().acc_policy, "auto")

    def test_acc_policy_valid_values(self):
        """All documented strategies are accepted verbatim."""
        for policy in ("accurate", "compact", "auto"):
            options = torch.nn.LinearCrossEntropyOptions(acc_policy=policy)
            self.assertEqual(options.acc_policy, policy)

    def test_acc_policy_invalid_raises(self):
        """Unknown strategy names and non-strings are rejected at construction."""
        with self.assertRaisesRegex(ValueError, "acc_policy"):
            torch.nn.LinearCrossEntropyOptions(acc_policy="fast")
        with self.assertRaisesRegex(ValueError, "acc_policy"):
            torch.nn.LinearCrossEntropyOptions(acc_policy=1)

    def test_acc_policy_auto_resolution(self):
        """_adjust resolves 'auto' per (device.type, dtype) per the documented rule."""
        # CPU low-precision input picks the accurate policy.
        resolved = torch.nn.LinearCrossEntropyOptions()._adjust(
            num_batches=8,
            in_features=4,
            num_classes=10,
            dtype=torch.float16,
            device=torch.device("cpu"),
        )
        self.assertEqual(resolved.acc_policy, "accurate")
        # CPU full-precision input stays compact.
        resolved = torch.nn.LinearCrossEntropyOptions()._adjust(
            num_batches=8,
            in_features=4,
            num_classes=10,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertEqual(resolved.acc_policy, "compact")
        # device=None forces the documented compact fallback.
        resolved = torch.nn.LinearCrossEntropyOptions()._adjust(
            num_batches=8,
            in_features=4,
            num_classes=10,
            dtype=torch.float16,
            device=None,
        )
        self.assertEqual(resolved.acc_policy, "compact")

    def test_acc_policy_auto_resolution_device_types(self):
        """Parseable device.type values resolve 'auto' to compact."""
        # 'meta' and 'cuda' device objects parse on every host with no
        # hardware; 'npu' only parses once torch_npu has registered the
        # backend, so it joins the list only when the probe detected NPU
        # (otherwise torch.device('npu') raises RuntimeError).
        device_types = ["meta", "cuda"]
        if self.device_name == "npu":
            device_types.append("npu")
        for device_type in device_types:
            resolved = torch.nn.LinearCrossEntropyOptions()._adjust(
                num_batches=8,
                in_features=4,
                num_classes=10,
                dtype=torch.bfloat16,
                device=torch.device(device_type),
            )
            self.assertEqual(resolved.acc_policy, "compact")


if __name__ == "__main__":
    run_tests()
