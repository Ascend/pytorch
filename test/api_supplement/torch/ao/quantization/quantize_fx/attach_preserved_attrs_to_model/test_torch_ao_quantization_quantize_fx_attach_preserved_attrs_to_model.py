# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.quantize_fx.attach_preserved_attrs_to_model 接口功能正确性
API 名称：torch.ao.quantization.quantize_fx.attach_preserved_attrs_to_model
API 签名：attach_preserved_attrs_to_model(model, preserved_attrs)

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                                  |
|---------------|--------------------------------------|-----------------------------------------------------------|
| 空/非空       | preserved_attrs 为 {} / 非空 / None  | 已覆盖：test_attach_empty_dict、                          |
|               |                                      | test_attach_multiple_attrs、                              |
|               |                                      | test_attach_none_dict                                     |
| 枚举选项      | 该接口无枚举参数                     | 不适用                                                    |
| 参数类型      | model 为 GraphModule；attrs 为 dict  | 已覆盖：test_attach_stores_attrs_in_meta、                |
|               |                                      | test_attach_requires_meta                                 |
| 传参与不传参  | 该接口无可选参数                     | 不适用                                                    |
| 等价类/边界值 | 单属性 / 多属性 / 覆盖已存在属性     | 已覆盖：test_attach_overwrites_existing_attr              |
| 正常传参场景  | 属性写入 model.meta 且可经           | 已覆盖：test_attach_stores_attrs_in_meta                  |
|               | model.attr 访问                      |                                                           |
| 异常传参场景  | 无 meta 的模块 / attrs 为 None       | 已覆盖：test_attach_requires_meta、test_attach_none_dict  |
| 混合设备类型  | 多 Tensor 异构设备输入               | 不适用，该接口仅操作 Python 属性，不涉及设备迁移          |

未覆盖项及原因：
- 枚举选项、传参与不传参：该接口无枚举参数与可选参数，无对应维度可覆盖。
- 混合设备类型：接口只把 Python 属性挂到 model.meta，不消费 Tensor，无设备执行路径，故不覆盖。

注：本接口为纯 Python 状态工具（框架类 A），无 CPU/NPU 数值输出，不做 CPU Golden 对照，
     按属性与 model.meta 结构断言。
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

from torch.ao.quantization.quantize_fx import attach_preserved_attrs_to_model
from torch.fx.graph_module import _USER_PRESERVED_ATTRIBUTES_KEY


def _make_graph_module():
    """Symbolically trace a tiny module to obtain a GraphModule with a meta dict."""

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    return torch.fx.symbolic_trace(_M().eval())


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestAttachPreservedAttrsToModelUt(TestCase):
    """Functional tests for torch.ao.quantization.quantize_fx.attach_preserved_attrs_to_model."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_attach_stores_attrs_in_meta(self):
        model = _make_graph_module()
        result = attach_preserved_attrs_to_model(model, {"my_attr": 123})
        # Returns None and stores the attributes both in meta and on the model.
        self.assertIsNone(result)
        self.assertEqual(model.my_attr, 123)
        self.assertIn(_USER_PRESERVED_ATTRIBUTES_KEY, model.meta)
        self.assertEqual(model.meta[_USER_PRESERVED_ATTRIBUTES_KEY], {"my_attr": 123})

    def test_attach_empty_dict(self):
        model = _make_graph_module()
        attach_preserved_attrs_to_model(model, {})
        self.assertEqual(model.meta[_USER_PRESERVED_ATTRIBUTES_KEY], {})
        self.assertFalse(hasattr(model, "my_attr"))

    def test_attach_multiple_attrs(self):
        model = _make_graph_module()
        attach_preserved_attrs_to_model(model, {"a": 1, "b": "two"})
        self.assertEqual(model.a, 1)
        self.assertEqual(model.b, "two")
        self.assertEqual(model.meta[_USER_PRESERVED_ATTRIBUTES_KEY], {"a": 1, "b": "two"})

    def test_attach_overwrites_existing_attr(self):
        model = _make_graph_module()
        model.loss_scale = 1.0
        attach_preserved_attrs_to_model(model, {"loss_scale": 2.5})
        # The preserved attribute must win over the pre-existing one.
        self.assertEqual(model.loss_scale, 2.5)

    def test_attach_copies_attrs_dict(self):
        model = _make_graph_module()
        attrs = {"a": 1}
        attach_preserved_attrs_to_model(model, attrs)
        stored = model.meta[_USER_PRESERVED_ATTRIBUTES_KEY]
        # The stored dict is a shallow copy: mutating the caller's dict afterwards
        # must not add keys to the stored one.
        self.assertIsNot(stored, attrs)
        attrs["b"] = 2
        self.assertNotIn("b", stored)

    def test_attach_requires_meta(self):
        # A plain nn.Module has no .meta, so the call must fail rather than silently no-op.
        with self.assertRaises(AttributeError):
            attach_preserved_attrs_to_model(torch.nn.Linear(3, 3), {"a": 1})

    def test_attach_none_dict(self):
        model = _make_graph_module()
        with self.assertRaises(AttributeError):
            attach_preserved_attrs_to_model(model, None)


if __name__ == "__main__":
    run_tests()
