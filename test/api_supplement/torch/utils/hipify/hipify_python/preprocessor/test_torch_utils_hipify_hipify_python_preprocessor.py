# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.preprocessor 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.preprocessor
API 签名：preprocessor(output_directory, filepath, all_files, header_include_dirs, stats,
         hip_clang_launch, is_pytorch_extension, clean_ctx, show_progress) -> HipifyResult

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空 all_files 与非空源码文件列表      | 已覆盖（ignored_file、basic_preprocessor）         |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（无枚举类参数）                             |
| 参数类型         | 路径、列表、统计字典与清理器上下文   | 已覆盖（basic_preprocessor）                       |
| 传参与不传参     | 必填参数传入与缺参                   | 已覆盖（basic_preprocessor、missing_args_raises）  |
| 等价类/边界值    | 正常转换与忽略文件返回状态           | 已覆盖（basic_preprocessor、ignored_file）         |
| 正常传参场景     | CUDA 源码 hipify 转换并返回结果对象  | 已覆盖（basic_preprocessor、ignored_file）         |
| 异常传参场景     | 缺失必填参数抛出 TypeError           | 已覆盖（missing_args_raises）                      |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（纯源码预处理工具，无 Tensor 输入）         |

未覆盖项及原因：
- 枚举选项：该接口参数为路径、集合、统计字典与布尔开关，无枚举类参数。
- 混合设备类型：该接口为纯源码 hipify 预处理函数，不涉及 Tensor 或设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import tempfile
import unittest

import torch
from torch.utils.hipify.hipify_python import (
    CurrentState,
    GeneratedFileCleaner,
    HIPIFY_FINAL_RESULT,
    HipifyResult,
    preprocessor,
)

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
class TestHipifyPreprocessor(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.preprocessor."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self._saved_results = dict(HIPIFY_FINAL_RESULT)

    def tearDown(self):
        HIPIFY_FINAL_RESULT.clear()
        HIPIFY_FINAL_RESULT.update(self._saved_results)
        super().tearDown()

    def test_basic_preprocessor(self):
        """Preprocesses a CUDA file and returns a DONE HipifyResult with hipified code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "sample.cu")
            unix_src = src_path.replace(os.sep, "/")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write("void foo() { cudaMalloc(nullptr, 16); }\n")
            fin_path = os.path.abspath(src_path)
            HIPIFY_FINAL_RESULT[fin_path] = HipifyResult(CurrentState.INITIALIZED, fin_path)
            stats = {"unsupported_calls": [], "kernel_launches": []}
            with GeneratedFileCleaner(keep_intermediates=True) as clean_ctx:
                result = preprocessor(
                    tmpdir, unix_src, [unix_src], [], stats, False, True, clean_ctx, False
                )
            self.assertIsInstance(result, HipifyResult)
            self.assertEqual(result.current_state, CurrentState.DONE)
            self.assertEqual(result.status, "[ok]")
            self.assertTrue(os.path.exists(result.hipified_path))
            with open(result.hipified_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("hipMalloc", content)

    def test_ignored_file(self):
        """File absent from all_files returns an ignored HipifyResult with hipified_path=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "ignored.cu")
            unix_src = src_path.replace(os.sep, "/")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write("void bar() { cudaMalloc(nullptr, 8); }\n")
            fin_path = os.path.abspath(src_path)
            HIPIFY_FINAL_RESULT[fin_path] = HipifyResult(CurrentState.INITIALIZED, fin_path)
            stats = {"unsupported_calls": [], "kernel_launches": []}
            with GeneratedFileCleaner(keep_intermediates=True) as clean_ctx:
                result = preprocessor(
                    tmpdir, unix_src, [], [], stats, False, True, clean_ctx, False
                )
            self.assertIsInstance(result, HipifyResult)
            self.assertIsNone(result.hipified_path)
            self.assertIn("ignored", result.status)

    def test_missing_args_raises(self):
        """Calling preprocessor without required arguments raises TypeError."""
        with self.assertRaises(TypeError):
            preprocessor()


if __name__ == "__main__":
    run_tests()
