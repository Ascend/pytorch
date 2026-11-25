# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.cpp_extension.load 接口功能正确性
API 名称：torch.utils.cpp_extension.load
API 签名：torch.utils.cpp_extension.load(name, sources, extra_cflags=None, ..., keep_intermediates=True)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 空源码与非空源码输入                       | 未覆盖（load 面向非空编译源码，mock 场景校验路径转发） |
| 枚举选项         | verbose/with_cuda/is_python_module 等开关  | 已覆盖（explicit_kwargs_forward、default_build_dir）   |
| 参数类型         | sources 为单字符串 str 与字符串列表 list   | 已覆盖（str_source_normalized、list_sources_forward）  |
| 传参与不传参     | 省略可选参数使用默认构建目录与显式传参     | 已覆盖（str_source_normalized、default_build_dir、explicit_kwargs_forward） |
| 等价类/边界值    | 字符串归一化为单元素列表与默认目录回退     | 已覆盖（str_source_normalized、default_build_dir）     |
| 正常传参场景     | 参数转发至 _jit_compile 并返回模块         | 已覆盖（str_source_normalized、list_sources_forward）  |
| 异常传参场景     | 缺少必填参数 name 或 sources               | 已覆盖（missing_args_raises）                          |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（JIT C++ 扩展编译加载入口，不接收 Tensor 输入） |

未覆盖项及原因：
- 空/非空：接口用于编译加载外部 C++ 源文件，单测通过 mock 校验非空字符串/列表参数规范化与转发逻辑。
- 混合设备类型：JIT C++ 扩展编译加载入口，不涉及 Tensor 输入与设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.utils.cpp_extension as cpp_ext

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
class TestCppExtensionLoad(TestCase):
    """Test cases for torch.utils.cpp_extension.load."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_str_source_normalized(self):
        """Normalizes a single string source into a one-element list for _jit_compile."""
        fake_mod = MagicMock(name="loaded_module")
        with patch.object(cpp_ext, "_jit_compile", return_value=fake_mod) as mock_jit, \
             patch.object(cpp_ext, "_get_build_directory", return_value="/tmp/ext_dir"):
            ret = cpp_ext.load(name="my_ext", sources="ext.cpp")
        self.assertIs(ret, fake_mod)
        mock_jit.assert_called_once()
        args, kwargs = mock_jit.call_args
        self.assertEqual(args[0], "my_ext")
        self.assertEqual(args[1], ["ext.cpp"])
        self.assertEqual(args[7], "/tmp/ext_dir")
        self.assertEqual(kwargs.get("keep_intermediates"), True)

    def test_list_sources_forward(self):
        """Forwards a list of source paths and custom build_directory to _jit_compile."""
        fake_mod = MagicMock(name="list_module")
        with patch.object(cpp_ext, "_jit_compile", return_value=fake_mod) as mock_jit:
            ret = cpp_ext.load(name="multi_ext", sources=["a.cpp", "b.cpp"], build_directory="/tmp/custom_build")
        self.assertIs(ret, fake_mod)
        args, _ = mock_jit.call_args
        self.assertEqual(args[0], "multi_ext")
        self.assertEqual(args[1], ["a.cpp", "b.cpp"])
        self.assertEqual(args[7], "/tmp/custom_build")

    def test_explicit_kwargs_forward(self):
        """Forwards explicit flags and compile options unchanged to _jit_compile."""
        fake_mod = MagicMock(name="explicit_module")
        with patch.object(cpp_ext, "_jit_compile", return_value=fake_mod) as mock_jit:
            ret = cpp_ext.load(
                name="opt_ext", sources=["opt.cpp"], extra_cflags=["-O3"], build_directory="/tmp/opt_dir",
                verbose=True, with_cuda=False, is_python_module=False, keep_intermediates=False,
            )
        self.assertIs(ret, fake_mod)
        args, kwargs = mock_jit.call_args
        self.assertEqual(args[2], ["-O3"])
        self.assertEqual(args[7], "/tmp/opt_dir")
        self.assertTrue(args[8])
        self.assertFalse(args[9])
        self.assertFalse(args[11])
        self.assertFalse(kwargs.get("keep_intermediates"))

    def test_default_build_dir(self):
        """Resolves default build directory via _get_build_directory when omitted."""
        with patch.object(cpp_ext, "_jit_compile", return_value=None), \
             patch.object(cpp_ext, "_get_build_directory", return_value="/tmp/auto") as mock_dir:
            cpp_ext.load(name="auto_ext", sources=["a.cpp"], verbose=True)
        mock_dir.assert_called_once_with("auto_ext", True)

    def test_missing_args_raises(self):
        """Calling load without required name or sources raises TypeError."""
        with self.assertRaises(TypeError):
            cpp_ext.load()
        with self.assertRaises(TypeError):
            cpp_ext.load("only_name")


if __name__ == "__main__":
    run_tests()
