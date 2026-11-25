# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.is_special_file 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.is_special_file
API 签名：torch.utils.hipify.hipify_python.is_special_file(rel_filepath) -> bool

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空路径与有效相对路径                 | 已覆盖（empty_path、positive_paths）               |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（无枚举类参数）                             |
| 参数类型         | 相对路径字符串参数                   | 已覆盖（positive_paths）                           |
| 传参与不传参     | 必填参数传入与缺失                   | 已覆盖（positive_paths、missing_arg_raises）       |
| 等价类/边界值    | 命中规则、排除项与普通路径           | 已覆盖（positive_paths、linalg_blas_excluded、     |
|                  |                                      | negative_paths）                                   |
| 正常传参场景     | sparse/linalg 路径与普通路径判定     | 已覆盖（positive_paths、negative_paths、           |
|                  |                                      | linalg_blas_excluded）                             |
| 异常传参场景     | 绝对路径与缺参异常                   | 已覆盖（abs_path_raises、missing_arg_raises）      |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（纯路径字符串工具，无 Tensor 输入）         |

未覆盖项及原因：
- 枚举选项：该接口仅接收相对路径字符串，无枚举类参数。
- 混合设备类型：该接口为纯路径字符串判定函数，不涉及 Tensor 或设备迁移。

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
from torch.utils.hipify.hipify_python import is_special_file

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
class TestHipifyIsSpecialFile(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.is_special_file."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_positive_paths(self):
        """PyTorch sparse and linalg paths return True."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res1 = is_special_file("aten/src/ATen/native/sparse/cuda/Sparse.cu")
            res2 = is_special_file("aten/src/ATen/native/linalg/BatchLinearAlgebra.cpp")
        self.assertIs(res1, True)
        self.assertIs(res2, True)

    def test_linalg_blas_excluded(self):
        """Specific batchlinearalgebralibblas linalg path returns False."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res = is_special_file("aten/src/ATen/native/linalg/BatchLinearAlgebraLibBlas.cpp")
        self.assertIs(res, False)

    def test_negative_paths(self):
        """Regular PyTorch paths and non-PyTorch sparse paths return False."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res1 = is_special_file("aten/src/ATen/native/cuda/Add.cu")
            res2 = is_special_file("external/sparse/kernel.cu")
        self.assertIs(res1, False)
        self.assertIs(res2, False)

    def test_empty_path(self):
        """Empty relative path returns False."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res = is_special_file("")
        self.assertIs(res, False)

    def test_abs_path_raises(self):
        """Absolute file path raises AssertionError."""
        abs_path = os.path.abspath("aten/src/ATen/native/sparse/foo.cu")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with self.assertRaises(AssertionError):
                is_special_file(abs_path)

    def test_missing_arg_raises(self):
        """Calling is_special_file without arguments raises TypeError."""
        with self.assertRaises(TypeError):
            is_special_file()


if __name__ == "__main__":
    run_tests()
