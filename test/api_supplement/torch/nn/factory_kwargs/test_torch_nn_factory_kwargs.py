# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.factory_kwargs 接口功能正确性
API 名称：torch.nn.factory_kwargs
API 签名：factory_kwargs(kwargs: dict | None) -> dict

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | kwargs=None 与空 dict 均返回空 dict；嵌套 factory_kwargs 空/非空合并 | 已覆盖（none/empty/nested_merge） |
| 枚举选项 | device/dtype/memory_format 三个合法键逐一与组合传参 | 已覆盖（test_factory_kwargs_simple_keys/test_factory_kwargs_combined） |
| 参数类型 | 结果为 dict；传入 dict 与 None；非法键名抛 TypeError | 已覆盖（returns_dict/unexpected_key_raises） |
| 传参与不传参 | 直接传简单键、经 factory_kwargs 键嵌套传参两条路径 | 已覆盖（test_factory_kwargs_simple_keys/test_factory_kwargs_nested_merge） |
| 等价类/边界值 | 输入 dict（含嵌套 dict）调用后原对象不被修改 | 已覆盖（test_factory_kwargs_input_not_mutated） |
| 正常传参场景 | 结果可直接展开传给 torch.empty 并在设备上生效 | 已覆盖（test_factory_kwargs_result_usable_with_factory） |
| 异常传参场景 | 未知键抛 TypeError(unexpected kwargs)；键冲突抛 TypeError(specified twice) | 已覆盖（unexpected_key/conflict） |
| 混合设备类型 | device 键使用当前设备（NPU/CUDA）真实建张量验证设备归属 | 已覆盖（test_factory_kwargs_result_usable_with_factory） |

未覆盖项及原因：
- memory_format=torch.channels_last 的真实建张量场景未覆盖：NPU 平台对 channels_last
  内存格式支持不完整（torch_npu clone/contiguous 不支持），仅断言该键在结果 dict 中透传，
  不在设备上实际创建 channels_last 张量。

注意：factory_kwargs 为纯 Python 工具（框架 A 类），按返回值内容、副作用与异常路径断言；
     唯一涉及设备的用例为把结果展开传给 torch.empty 验证真实设备分配。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"

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
class TestFactoryKwargs(TestCase):
    """Comprehensive cases for torch.nn.factory_kwargs (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_factory_kwargs_none(self):
        """factory_kwargs(None) returns an empty dict."""
        self.assertEqual(torch.nn.factory_kwargs(None), {})

    def test_factory_kwargs_empty(self):
        """An empty dict input returns an empty dict."""
        self.assertEqual(torch.nn.factory_kwargs({}), {})

    def test_factory_kwargs_returns_dict(self):
        """The result is a plain dict for dict inputs."""
        result = torch.nn.factory_kwargs({"dtype": torch.float32})
        self.assertIsInstance(result, dict)

    def test_factory_kwargs_simple_keys(self):
        """Each simple key passes through with its value unchanged."""
        result = torch.nn.factory_kwargs({"device": self.device_name})
        self.assertEqual(result, {"device": self.device_name})
        result = torch.nn.factory_kwargs({"dtype": torch.float64})
        self.assertEqual(result, {"dtype": torch.float64})
        result = torch.nn.factory_kwargs({"memory_format": torch.contiguous_format})
        self.assertEqual(result, {"memory_format": torch.contiguous_format})
        result = torch.nn.factory_kwargs({"memory_format": torch.channels_last})
        self.assertEqual(result, {"memory_format": torch.channels_last})

    def test_factory_kwargs_combined(self):
        """All three simple keys combine into one canonical dict."""
        kwargs = {
            "device": self.device_name,
            "dtype": torch.float32,
            "memory_format": torch.contiguous_format,
        }
        result = torch.nn.factory_kwargs(kwargs)
        self.assertEqual(result, kwargs)

    def test_factory_kwargs_nested_merge(self):
        """A nested factory_kwargs dict merges with the simple keys."""
        result = torch.nn.factory_kwargs(
            {"device": self.device_name,
             "factory_kwargs": {"dtype": torch.float64}})
        self.assertEqual(result, {"device": self.device_name,
                                  "dtype": torch.float64})
        self.assertNotIn("factory_kwargs", result)

    def test_factory_kwargs_nested_only(self):
        """A nested-only input yields exactly the nested dict content."""
        result = torch.nn.factory_kwargs({"factory_kwargs": {"dtype": torch.float16}})
        self.assertEqual(result, {"dtype": torch.float16})

    def test_factory_kwargs_unexpected_key_raises(self):
        """An unrecognized key raises TypeError mentioning the offending keys."""
        with self.assertRaisesRegex(TypeError, "unexpected kwargs"):
            torch.nn.factory_kwargs({"stride": 2})

    def test_factory_kwargs_conflict_raises(self):
        """The same key in kwargs and the nested dict raises TypeError."""
        with self.assertRaisesRegex(TypeError, "device specified twice"):
            torch.nn.factory_kwargs(
                {"device": self.device_name,
                 "factory_kwargs": {"device": "cpu"}})
        with self.assertRaisesRegex(TypeError, "dtype specified twice"):
            torch.nn.factory_kwargs(
                {"dtype": torch.float32,
                 "factory_kwargs": {"dtype": torch.float64}})

    def test_factory_kwargs_input_not_mutated(self):
        """The input dict and nested dict are left untouched by the call."""
        kwargs = {"device": self.device_name,
                  "factory_kwargs": {"dtype": torch.float64}}
        nested = kwargs["factory_kwargs"]
        torch.nn.factory_kwargs(kwargs)
        self.assertEqual(kwargs, {"device": self.device_name,
                                  "factory_kwargs": {"dtype": torch.float64}})
        self.assertEqual(nested, {"dtype": torch.float64})
        self.assertIs(kwargs["factory_kwargs"], nested)

    def test_factory_kwargs_result_usable_with_factory(self):
        """The canonical dict drives torch.empty allocation on the device."""
        result = torch.nn.factory_kwargs(
            {"device": self.device_name, "dtype": torch.float32})
        tensor = torch.empty((2, 3), **result)
        self.assertEqual(tensor.device.type, self.device_name)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tuple(tensor.shape), (2, 3))


if __name__ == "__main__":
    run_tests()
