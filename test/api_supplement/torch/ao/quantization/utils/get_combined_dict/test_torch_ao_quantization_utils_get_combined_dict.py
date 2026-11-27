# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.utils.get_combined_dict 接口功能正确性
API 名称：torch.ao.quantization.utils.get_combined_dict
API 签名：get_combined_dict(default_dict, additional_dict) -> dict

覆盖维度表：
| 覆盖维度         | 说明                                              | 覆盖情况                                                     |
|------------------|---------------------------------------------------|--------------------------------------------------------------|
| 空/非空          | default_dict / additional_dict 为空 / 非空        | 已覆盖：test_empty_default、test_empty_additional             |
| 枚举选项         | 该接口无枚举参数                                  | 不适用                                                       |
| 参数类型         | 两个参数均为 dict                                 | 已覆盖：test_merge_overrides_duplicate_keys                   |
| 传参与不传参     | 该接口无可选参数                                  | 不适用                                                       |
| 等价类/边界值    | 无键冲突 / 有键冲突（后者覆盖前者）               | 已覆盖：test_merge_overrides_duplicate_keys                   |
| 正常传参场景     | 返回合并后的新字典                                | 已覆盖：test_merge_overrides_duplicate_keys                   |
| 异常传参场景     | 传入非 dict（list）触发 AttributeError            | 已覆盖：test_non_dict_raises                                 |
| 混合设备类型     | 多 Tensor 异构设备输入                            | 不适用，该接口仅合并字典，不涉及设备                          |

未覆盖项及原因：
- 枚举选项、传参与不传参：接口无枚举/可选参数，无对应维度可覆盖。
- 混合设备类型：纯字典工具，无 Tensor 执行路径，故不覆盖。

注：该 API 为纯 Python 字典工具（框架类 A），无 CPU/NPU 数值输出，按字典内容断言。
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

from torch.ao.quantization.utils import get_combined_dict


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetCombinedDictUt(TestCase):
    """Functional tests for torch.ao.quantization.utils.get_combined_dict."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_merge_overrides_duplicate_keys(self):
        default = {"a": 1, "b": 1}
        additional = {"b": 2, "c": 3}
        self.assertEqual(get_combined_dict(default, additional), {"a": 1, "b": 2, "c": 3})

    def test_empty_default(self):
        self.assertEqual(get_combined_dict({}, {"x": 1}), {"x": 1})

    def test_empty_additional(self):
        self.assertEqual(get_combined_dict({"x": 1}, {}), {"x": 1})

    def test_returns_new_dict(self):
        default = {"a": 1}
        additional = {"b": 2}
        result = get_combined_dict(default, additional)
        # The inputs must not be mutated and the result is a fresh dict.
        self.assertEqual(default, {"a": 1})
        self.assertEqual(additional, {"b": 2})
        self.assertIsNot(result, default)
        self.assertIsNot(result, additional)

    def test_non_dict_raises(self):
        with self.assertRaises(AttributeError):
            get_combined_dict([], {})


if __name__ == "__main__":
    run_tests()
