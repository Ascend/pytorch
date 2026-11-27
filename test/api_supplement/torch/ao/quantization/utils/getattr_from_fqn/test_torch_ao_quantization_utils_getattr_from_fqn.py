# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.utils.getattr_from_fqn 接口功能正确性
API 名称：torch.ao.quantization.utils.getattr_from_fqn
API 签名：getattr_from_fqn(obj, fqn)

覆盖维度表：
| 覆盖维度         | 说明                                              | 覆盖情况                                                     |
|------------------|---------------------------------------------------|--------------------------------------------------------------|
| 空/非空          | fqn 为空串 / 非空串                               | 已覆盖：test_empty_fqn_raises、test_single_attribute          |
| 枚举选项         | 该接口无枚举参数                                  | 不适用                                                       |
| 参数类型         | obj 为 nn.Module / 普通对象，fqn 为 str           | 已覆盖：test_nested_module_attribute、test_scalar_attribute   |
| 传参与不传参     | 该接口无可选参数                                  | 不适用                                                       |
| 等价类/边界值    | 单层 / 多层（含三层）属性路径                     | 已覆盖：test_single_attribute、test_three_level_nesting       |
| 正常传参场景     | 返回路径末端对象                                  | 已覆盖：test_nested_module_attribute                         |
| 异常传参场景     | 无效路径 / 空 fqn 触发 AttributeError             | 已覆盖：test_invalid_fqn_raises、test_empty_fqn_raises        |
| 混合设备类型     | 多 Tensor 异构设备输入                            | 不适用，该接口只按名字取值，不涉及设备                        |

未覆盖项及原因：
- 枚举选项、传参与不传参：接口无枚举/可选参数，无对应维度可覆盖。
- 混合设备类型：纯属性取值工具，无 Tensor 执行路径，故不覆盖。

注：该 API 为纯 Python 属性工具（框架类 A），无 CPU/NPU 数值输出，按对象身份与异常断言。
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

from torch.ao.quantization.utils import getattr_from_fqn


class _Sub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = torch.nn.Linear(4, 4)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.sub = _Sub()
        self.count = 7

    def forward(self, x):
        return self.linear(x)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetattrFromFqnUt(TestCase):
    """Functional tests for torch.ao.quantization.utils.getattr_from_fqn."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_single_attribute(self):
        model = _Model()
        self.assertIs(getattr_from_fqn(model, "linear"), model.linear)

    def test_nested_module_attribute(self):
        model = _Model()
        self.assertIs(getattr_from_fqn(model, "sub.inner"), model.sub.inner)

    def test_three_level_nesting(self):
        model = _Model()
        self.assertIs(getattr_from_fqn(model, "sub.inner.weight"), model.sub.inner.weight)

    def test_scalar_attribute(self):
        model = _Model()
        self.assertEqual(getattr_from_fqn(model, "count"), 7)

    def test_invalid_fqn_raises(self):
        model = _Model()
        with self.assertRaises(AttributeError):
            getattr_from_fqn(model, "missing.attr")

    def test_empty_fqn_raises(self):
        model = _Model()
        with self.assertRaises(AttributeError):
            getattr_from_fqn(model, "")


if __name__ == "__main__":
    run_tests()
