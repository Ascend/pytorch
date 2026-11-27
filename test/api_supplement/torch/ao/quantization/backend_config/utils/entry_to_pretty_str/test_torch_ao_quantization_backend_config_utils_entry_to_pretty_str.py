# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.backend_config.utils.entry_to_pretty_str 接口功能正确性
API 名称：torch.ao.quantization.backend_config.utils.entry_to_pretty_str
API 签名：entry_to_pretty_str(entry) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 空 entry 与含各字段的非空 entry | 已覆盖：test_empty_entry_returns_empty_braces / test_pattern_entry_exact_string |
| 枚举选项 | 该接口无枚举参数 | 不适用 |
| 参数类型 | entry 为 dict，字段值可为 str/类/函数/嵌套结构 | 已覆盖：test_pattern_uses_human_readable_name |
| 传参与不传参 | 该接口无默认参数，entry 必传 | 不适用 |
| 等价类/边界值 | 单字段/多字段/字段顺序/dtype_configs 多元素 | 已覆盖：test_pattern_before_other_fields / test_dtype_configs_block |
| 正常传参场景 | 各字段正确格式化输出 | 已覆盖：test_dtype_configs_block / test_num_tensor_args_block / test_plain_fields_block |
| 异常传参场景 | 非 dict 输入无稳定契约异常 | 未覆盖，原因见下 |
| 混合设备类型 | 纯 Python 结构工具，无设备执行路径 | 不适用 |

未覆盖项及原因：
- 异常传参场景：entry_to_pretty_str 对任意 dict 均只做字符串拼接，非 dict 输入由内部迭代/成员判断触发
  上层类型错误，行为依赖 Python 版本且无稳定契约，故不单列。
- 混合设备类型：该接口仅做字符串格式化，不涉及任何 Tensor 或设备，无 CPU/NPU 对照路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为纯 Python 结构工具（A 类），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回字符串内容做精确断言。
"""
import os
import unittest

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

import torch.nn
import torch.nn.functional
from torch.ao.quantization.backend_config.utils import entry_to_pretty_str


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEntryToPrettyStrUt(TestCase):
    """Functional tests for torch.ao.quantization.backend_config.utils.entry_to_pretty_str."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_empty_entry_returns_empty_braces(self):
        """An empty entry renders as a bare pair of braces."""
        self.assertEqual(entry_to_pretty_str({}), "{\n}")

    def test_pattern_entry_exact_string(self):
        """The 'pattern' field is emitted first with two-space indentation."""
        self.assertEqual(
            entry_to_pretty_str({"pattern": "torch.nn.Linear"}),
            "{\n  'pattern': torch.nn.Linear,\n}",
        )

    def test_pattern_before_other_fields(self):
        """'pattern' is emitted first even when it is not the first dict key."""
        entry = {"root_module": "torch.nn.Linear", "pattern": "torch.add"}
        expected = "{\n  'pattern': torch.add,\n  'root_module': torch.nn.Linear,\n}"
        self.assertEqual(entry_to_pretty_str(entry), expected)

    def test_dtype_configs_block(self):
        """'dtype_configs' uses the nested pretty-printed block form."""
        entry = {
            "dtype_configs": [
                {"input_dtype": "torch.quint8", "output_dtype": "torch.quint8"},
            ]
        }
        expected = (
            "{\n"
            "  'dtype_configs': [\n"
            "    {\n"
            "      'input_dtype': torch.quint8,\n"
            "      'output_dtype': torch.quint8,\n"
            "    },\n"
            "  ],\n"
            "}"
        )
        self.assertEqual(entry_to_pretty_str(entry), expected)

    def test_num_tensor_args_block(self):
        """'num_tensor_args_to_observation_type' uses the digit-keyed block form."""
        entry = {
            "num_tensor_args_to_observation_type": {
                0: "OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT",
                2: "OUTPUT_SHARE_OBSERVER_WITH_INPUT",
            }
        }
        expected = (
            "{\n"
            "  'num_tensor_args_to_observation_type': {\n"
            "    0: OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,\n"
            "    2: OUTPUT_SHARE_OBSERVER_WITH_INPUT,\n"
            "  },\n"
            "}"
        )
        self.assertEqual(entry_to_pretty_str(entry), expected)

    def test_plain_fields_block(self):
        """Fields outside the custom-handled set are emitted verbatim."""
        entry = {
            "observation_type": "OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT",
            "root_module": "torch.nn.Linear",
        }
        expected = (
            "{\n"
            "  'observation_type': OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,\n"
            "  'root_module': torch.nn.Linear,\n"
            "}"
        )
        self.assertEqual(entry_to_pretty_str(entry), expected)

    def test_pattern_uses_human_readable_name(self):
        """A boolean_dispatch op renders as its human readable dotted name."""
        result = entry_to_pretty_str({"pattern": torch.nn.functional.max_pool2d})
        self.assertIn("'pattern': torch.nn.functional.max_pool2d,", result)
        self.assertNotIn("boolean_dispatch", result)


if __name__ == "__main__":
    run_tests()
