# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.is_cusparse_file 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.is_cusparse_file
API 签名：torch.utils.hipify.hipify_python.is_cusparse_file(rel_filepath)

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                              |
|---------------|--------------------------------------|-------------------------------------------------------|
| 空/非空       | 空路径字符串与非空相对路径           | 已覆盖（empty_path、positive_paths）                  |
| 枚举选项      | 枚举型参数选项                       | 未覆盖（仅接收相对路径 rel_filepath，无枚举选项）     |
| 参数类型      | str 相对路径与 bool 返回值           | 已覆盖（positive_paths、negative_paths）              |
| 传参与不传参  | 传入 rel_filepath 与缺参调用         | 已覆盖（positive_paths、missing_arg_raises）          |
| 等价类/边界值 | 大小写不敏感、非稀疏/非 PyTorch 路径 | 已覆盖（case_insensitive、negative_paths、empty_path） |
| 正常传参场景  | 判定 PyTorch cuSPARSE 相关文件路径   | 已覆盖（positive_paths、case_insensitive、negative_paths） |
| 异常传参场景  | 绝对路径 AssertionError 与缺参异常   | 已覆盖（abs_path_raises、missing_arg_raises）         |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合        | 未覆盖（纯路径字符串判定函数，无 Tensor 输入）        |

未覆盖项及原因：
- 枚举选项：接口仅接收单个相对路径字符串参数 rel_filepath，无枚举类参数。
- 混合设备类型：接口仅根据相对文件路径规则返回布尔值，无 Tensor 输入，不涉及设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest
import warnings

import torch
from torch.utils.hipify.hipify_python import is_cusparse_file

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
class TestHipifyIsCusparseFile(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.is_cusparse_file."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_positive_paths(self):
        """PyTorch paths containing sparse/cusparse return True."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res1 = is_cusparse_file("aten/src/ATen/native/sparse/cuda/SparseCUDATensor.cu")
            res2 = is_cusparse_file("torch/csrc/cusparse_utils.cpp")
        self.assertIs(res1, True)
        self.assertIs(res2, True)

    def test_case_insensitive(self):
        """Sparse keyword matching is case-insensitive."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res = is_cusparse_file("aten/src/ATen/native/SparseTensor.cu")
        self.assertIs(res, True)

    def test_negative_paths(self):
        """Non-sparse PyTorch paths and non-PyTorch paths return False."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res1 = is_cusparse_file("aten/src/ATen/native/cuda/AddKernel.cu")
            res2 = is_cusparse_file("external/cusparse/kernel.cu")
            res3 = is_cusparse_file("aten/src/ATen/core/sparse_core.cpp")
        self.assertIs(res1, False)
        self.assertIs(res2, False)
        self.assertIs(res3, False)

    def test_empty_path(self):
        """Empty relative path returns False."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res = is_cusparse_file("")
        self.assertIs(res, False)

    def test_abs_path_raises(self):
        """Absolute file path raises AssertionError."""
        abs_path = os.path.abspath("aten/src/ATen/native/sparse/foo.cu")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with self.assertRaises(AssertionError):
                is_cusparse_file(abs_path)

    def test_missing_arg_raises(self):
        """Calling is_cusparse_file without arguments raises TypeError."""
        with self.assertRaises(TypeError):
            is_cusparse_file()


if __name__ == "__main__":
    run_tests()
