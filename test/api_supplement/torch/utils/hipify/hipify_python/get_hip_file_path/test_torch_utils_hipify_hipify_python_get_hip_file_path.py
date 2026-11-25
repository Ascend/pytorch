# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.get_hip_file_path 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.get_hip_file_path
API 签名：get_hip_file_path(rel_filepath, is_pytorch_extension=False)

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                              |
|---------------|--------------------------------------|-------------------------------------------------------|
| 空/非空       | 空路径与非空相对路径转换             | 未覆盖（空字符串不代表有效相对源文件路径）            |
| 枚举选项      | is_pytorch_extension=False/True 模式 | 已覆盖（non_cuda_dir_default、extension_mode）        |
| 参数类型      | str 相对路径与 bool 扩展模式标志     | 已覆盖（cuda_dir_and_cu_ext、extension_mode）         |
| 传参与不传参  | is_pytorch_extension 默认与显式传参  | 已覆盖（cuda_dir_and_cu_ext、non_cuda_dir_default）   |
| 等价类/边界值 | 非 cuda 目录、in-place 路径与 THC    | 已覆盖（non_cuda_dir_default、inplace_torch_path、thc_prefix_replacement） |
| 正常传参场景  | CUDA/THC 目录与 .cu 扩展名路径映射   | 已覆盖（cuda_dir_and_cu_ext、extension_mode、inplace_torch_path） |
| 异常传参场景  | 绝对路径 AssertionError 与缺参异常   | 已覆盖（abs_path_raises、missing_arg_raises）         |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合        | 未覆盖（纯相对文件路径映射函数，无 Tensor 输入）      |

未覆盖项及原因：
- 空/非空：get_hip_file_path 面向有效相对源文件路径，空字符串无对应目录与文件名语义。
- 混合设备类型：接口仅对相对文件路径字符串进行规则变换，无 Tensor 输入，不涉及设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.hipify.hipify_python import get_hip_file_path

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
class TestHipifyGetHipFilePath(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.get_hip_file_path."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_cuda_dir_and_cu_ext(self):
        """Replaces cuda/CUDA directory and filename tokens and .cu extension."""
        result = get_hip_file_path("aten/src/ATen/cuda/FooCUDA.cu")
        self.assertEqual(result, os.path.join("aten/src/ATen/hip", "FooHIP.hip"))

    def test_non_cuda_dir_default(self):
        """Non-extension path without cuda dir inserts a 'hip' parent subdirectory."""
        result = get_hip_file_path("foo/bar.cu", is_pytorch_extension=False)
        self.assertEqual(result, os.path.join("foo", "hip", "bar.hip"))

    def test_extension_mode(self):
        """Extension mode appends '_hip' if unchanged, or only changes .cu to .hip."""
        cpp_res = get_hip_file_path("ext/ops.cpp", is_pytorch_extension=True)
        self.assertEqual(cpp_res, os.path.join("ext", "ops_hip.cpp"))

        cu_res = get_hip_file_path("ext/kernel.cu", is_pytorch_extension=True)
        self.assertEqual(cu_res, os.path.join("ext", "kernel.hip"))

    def test_inplace_torch_path(self):
        """In-place torch/ paths are returned unchanged when not an extension."""
        rel = "torch/csrc/cuda/Module.cpp"
        self.assertEqual(get_hip_file_path(rel, is_pytorch_extension=False), rel)

    def test_thc_prefix_replacement(self):
        """THC directory and filename prefixes are converted to THH."""
        result = get_hip_file_path("aten/src/THC/THCTensor.cu")
        self.assertEqual(result, os.path.join("aten/src/THH", "THHTensor.hip"))

    def test_abs_path_raises(self):
        """Absolute file path violates contract and raises AssertionError."""
        abs_path = os.path.abspath("foo/bar.cu")
        with self.assertRaises(AssertionError):
            get_hip_file_path(abs_path)

    def test_missing_arg_raises(self):
        """Calling get_hip_file_path without rel_filepath raises TypeError."""
        with self.assertRaises(TypeError):
            get_hip_file_path()


if __name__ == "__main__":
    run_tests()
