# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.str2bool 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.str2bool
API 签名：torch.utils.hipify.hipify_python.str2bool(v: str) -> bool

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空字符串与非空布尔字符串             | 已覆盖（invalid_strings_raise、truthy_strings）    |
| 枚举选项         | 真值/假值字符串枚举集合              | 已覆盖（truthy_strings、falsy_strings、            |
|                  |                                      | mixed_case_strings）                               |
| 参数类型         | 字符串与非字符串（int/None）         | 已覆盖（truthy_strings、                           |
|                  |                                      | non_string_raises_attribute_error）                |
| 传参与不传参     | 必填参数传入与缺参                   | 已覆盖（truthy_strings、missing_arg_raises）       |
| 等价类/边界值    | 混合大小写、尾随空格与未知字符串     | 已覆盖（mixed_case_strings、invalid_strings_raise）|
| 正常传参场景     | 真值与假值字符串解析为 bool          | 已覆盖（truthy_strings、falsy_strings、            |
|                  |                                      | mixed_case_strings）                               |
| 异常传参场景     | 非法字符串、非字符串与缺参异常       | 已覆盖（invalid_strings_raise、missing_arg_raises、|
|                  |                                      | non_string_raises_attribute_error）                |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（纯命令行布尔解析工具，无 Tensor 输入）     |

未覆盖项及原因：
- 混合设备类型：该接口为纯字符串到布尔值解析函数，不涉及 Tensor 或设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import argparse
import os
import unittest

import torch
from torch.utils.hipify.hipify_python import str2bool

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe device only when RUN_FLAG is enabled.
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
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestHipifyStr2Bool(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.str2bool."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_truthy_strings(self):
        """All documented truthy strings return True regardless of case."""
        for value in ("yes", "true", "t", "y", "1", "True", "YES"):
            self.assertIs(str2bool(value), True)

    def test_falsy_strings(self):
        """All documented falsy strings return False regardless of case."""
        for value in ("no", "false", "f", "n", "0", "False", "NO"):
            self.assertIs(str2bool(value), False)

    def test_mixed_case_strings(self):
        """Mixed-case truthy and falsy strings are normalized via lower()."""
        self.assertIs(str2bool("YeS"), True)
        self.assertIs(str2bool("TrUe"), True)
        self.assertIs(str2bool("No"), False)
        self.assertIs(str2bool("FaLsE"), False)

    def test_non_string_raises_attribute_error(self):
        """Passing non-string value without lower() raises AttributeError."""
        with self.assertRaises(AttributeError):
            str2bool(1)
        with self.assertRaises(AttributeError):
            str2bool(None)

    def test_invalid_strings_raise(self):
        """Unrecognized or empty strings raise argparse.ArgumentTypeError."""
        for bad in ("maybe", "2", "", "true "):
            with self.assertRaises(argparse.ArgumentTypeError):
                str2bool(bad)

    def test_missing_arg_raises(self):
        """Calling str2bool without arguments raises TypeError."""
        with self.assertRaises(TypeError):
            str2bool()


if __name__ == "__main__":
    run_tests()
