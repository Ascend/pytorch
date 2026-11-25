# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.cpp_extension.SyclExtension 接口功能正确性
API 名称：torch.utils.cpp_extension.SyclExtension
API 签名：torch.utils.cpp_extension.SyclExtension(name, sources, *args, **kwargs)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 空 sources 列表与非空源码列表              | 已覆盖（empty_sources、basic_construction）            |
| 枚举选项         | py_limited_api 布尔开关（默认 False/True） | 已覆盖（basic_construction、py_limited_api_flag）      |
| 参数类型         | 字符串 name 与列表形式的 sources/路径/标志 | 已覆盖（basic_construction、custom_kwargs）            |
| 传参与不传参     | 仅必填参数与显式传入自定义 kwargs          | 已覆盖（basic_construction、custom_kwargs、py_limited_api_flag） |
| 等价类/边界值    | 空源码边界与 py_limited_api 裁剪链接库     | 已覆盖（empty_sources、py_limited_api_flag）           |
| 正常传参场景     | 构造含默认/自定义路径与编译参数的扩展对象  | 已覆盖（basic_construction、custom_kwargs）            |
| 异常传参场景     | 缺少 name 或 sources 必填参数              | 已覆盖（missing_args_raises）                          |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（构建 setuptools.Extension 配置，无 Tensor 输入） |

未覆盖项及原因：
- 混合设备类型：接口用于构造 C++/SYCL setuptools 扩展配置对象，不接收 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import setuptools
import torch
from torch.utils.cpp_extension import SyclExtension

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

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
class TestSyclExtension(TestCase):
    """Test cases for torch.utils.cpp_extension.SyclExtension."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def _mock_sycl_paths(self):
        from unittest.mock import patch
        return (
            patch("torch.utils.cpp_extension.include_paths", return_value=["/mock/include"]),
            patch("torch.utils.cpp_extension.library_paths", return_value=["/mock/lib"]),
        )

    def test_basic_construction(self):
        """Constructs a setuptools.Extension with default SYCL include/library settings."""
        p_inc, p_lib = self._mock_sycl_paths()
        with p_inc, p_lib:
            ext = SyclExtension(name="my_sycl_ext", sources=["ext.cpp", "kernel.sycl"])
        self.assertIsInstance(ext, setuptools.Extension)
        self.assertEqual(ext.name, "my_sycl_ext")
        self.assertEqual(ext.sources, ["ext.cpp", "kernel.sycl"])
        self.assertEqual(ext.language, "c++")
        self.assertIn("/mock/include", ext.include_dirs)
        self.assertIn("/mock/lib", ext.library_dirs)
        for lib in ("c10", "c10_xpu", "torch", "torch_cpu", "sycl", "torch_python", "torch_xpu"):
            self.assertIn(lib, ext.libraries)

    def test_custom_kwargs(self):
        """Merges user-provided include_dirs, library_dirs, libraries, and compile args."""
        p_inc, p_lib = self._mock_sycl_paths()
        with p_inc, p_lib:
            ext = SyclExtension(
                name="custom_ext", sources=["a.cpp"], include_dirs=["/custom/include"],
                library_dirs=["/custom/lib"], libraries=["custom_lib"], extra_compile_args=["-O3", "-std=c++20"],
            )
        self.assertIn("/custom/include", ext.include_dirs)
        self.assertIn("/custom/lib", ext.library_dirs)
        self.assertIn("custom_lib", ext.libraries)
        self.assertIn("sycl", ext.libraries)
        self.assertEqual(ext.extra_compile_args, ["-O3", "-std=c++20"])

    def test_py_limited_api_flag(self):
        """Omits torch_python from libraries when py_limited_api=True."""
        p_inc, p_lib = self._mock_sycl_paths()
        with p_inc, p_lib:
            ext = SyclExtension(name="limited_ext", sources=["a.cpp"], py_limited_api=True)
        self.assertNotIn("torch_python", ext.libraries)
        self.assertIn("torch_xpu", ext.libraries)

    def test_empty_sources(self):
        """Accepts an empty sources list and retains extension metadata."""
        p_inc, p_lib = self._mock_sycl_paths()
        with p_inc, p_lib:
            ext = SyclExtension(name="empty_ext", sources=[])
        self.assertEqual(ext.name, "empty_ext")
        self.assertEqual(ext.sources, [])
        self.assertEqual(ext.language, "c++")

    def test_missing_args_raises(self):
        """Calling SyclExtension without required arguments raises TypeError."""
        with self.assertRaises(TypeError):
            SyclExtension()
        with self.assertRaises(TypeError):
            SyclExtension("only_name")


if __name__ == "__main__":
    run_tests()
