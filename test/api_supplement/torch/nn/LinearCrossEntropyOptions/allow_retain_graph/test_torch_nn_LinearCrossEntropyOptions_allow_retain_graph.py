# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.LinearCrossEntropyOptions.allow_retain_graph 接口功能正确性
API 名称：torch.nn.LinearCrossEntropyOptions.allow_retain_graph
API 签名：allow_retain_graph: bool = False（dataclass 字段，构造参数 allow_retain_graph=...）

说明：allow_retain_graph 是 LinearCrossEntropyOptions 的反向缓冲保留开关，仅作用于
reduction='mean'/'sum' 的分块路径：False（默认）时反向就地消耗预计算梯度缓冲，第二次
.backward() 抛 RuntimeError；True 时保留缓冲，可连续反向，代价为每次调用多一份梯度尺寸
分配。reduction='none' 的反向无条件支持二次反向，忽略本字段。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 默认 False 与显式 True | 已覆盖（_default_false/_explicit_true） |
| 枚举选项 | False/True 两个取值 | 已覆盖（_default_false/_explicit_true） |
| 参数类型 | 布尔字段保持 | 已覆盖（_explicit_true） |
| 传参与不传参 | 省略（默认 False）与显式传参 | 已覆盖（_default_false/_explicit_true） |
| 等价类/边界值 | replace 派生与 _adjust 解析后字段保持 | 已覆盖（_replace_and_adjust_preserved） |
| 正常传参场景 | 构造、replace、_adjust 全链路字段不丢 | 已覆盖（_replace_and_adjust_preserved） |
| 异常传参场景 | 本字段构造期无校验（任意对象被接受），无稳定异常契约 | 未覆盖（构造不校验取值，见下） |
| 混合设备类型 | 纯 Python 布尔字段，无设备执行路径 | 未覆盖（行为契约无设备差异，见下） |

未覆盖项及原因：
- 二次反向的行为契约（False 抛 RuntimeError / True 可连续反向且梯度累加）属于
  F.linear_cross_entropy 分块路径的执行行为，由
  torch.nn.functional.linear_cross_entropy 的功能用例
  （test_linear_cross_entropy_chunked_second_backward）在设备上真实执行验证；
  本文件看护容器内该字段的存取与传递契约。
- 异常传参：allow_retain_graph 不做构造期类型校验（truthy/falsy 均被接受），无稳定异常
  路径可断言。
- 混合设备类型：布尔字段无设备语义。

注意：纯 Python 字段契约按取值与传递结果断言；无浮点 Tensor 输出，不做 CPU Golden
     数值对照。字段行为语义以 PyTorch 2.15.0 源码为准。
"""
import dataclasses
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
class TestLinearCrossEntropyOptionsAllowRetainGraph(TestCase):
    """Test cases for the allow_retain_graph field of LinearCrossEntropyOptions."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_allow_retain_graph_default_false(self):
        """Omitted allow_retain_graph resolves to the documented False default."""
        self.assertFalse(torch.nn.LinearCrossEntropyOptions().allow_retain_graph)

    def test_allow_retain_graph_explicit_true(self):
        """Explicit True is stored verbatim."""
        options = torch.nn.LinearCrossEntropyOptions(allow_retain_graph=True)
        self.assertTrue(options.allow_retain_graph)

    def test_allow_retain_graph_replace_and_adjust_preserved(self):
        """replace() and _adjust() both carry the field through unchanged."""
        original = torch.nn.LinearCrossEntropyOptions(allow_retain_graph=True)
        derived = dataclasses.replace(original, batch_chunk_size=3)
        self.assertTrue(derived.allow_retain_graph)
        resolved = original._adjust(
            num_batches=8,
            in_features=4,
            num_classes=10,
            dtype=torch.float32,
            device=None,
        )
        self.assertTrue(resolved.allow_retain_graph)
        resolved_default = torch.nn.LinearCrossEntropyOptions()._adjust(
            num_batches=8,
            in_features=4,
            num_classes=10,
            dtype=torch.float32,
            device=None,
        )
        self.assertFalse(resolved_default.allow_retain_graph)


if __name__ == "__main__":
    run_tests()
