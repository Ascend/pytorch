# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.hipify 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.hipify
API 签名：hipify(project_directory, show_detailed=False, extensions=..., output_directory='', show_progress=True, ...)

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                              |
|---------------|--------------------------------------|-------------------------------------------------------|
| 空/非空       | 空项目目录与含 .cu 文件的非空目录    | 已覆盖（empty_project_dir、basic_cu_conversion）      |
| 枚举选项      | 多布尔开关与扩展名过滤组合           | 未覆盖（仅传 show_progress=False，未穷举全部选项组合） |
| 参数类型      | str 目录路径与 bool 进度条开关       | 已覆盖（basic_cu_conversion、empty_project_dir）      |
| 传参与不传参  | 可选参数省略/显式传参与缺少必需参数  | 已覆盖（basic_cu_conversion、nonexistent_dir_exits、missing_arg_raises） |
| 等价类/边界值 | 空目录返回空字典与不存在目录退出     | 已覆盖（empty_project_dir、nonexistent_dir_exits）    |
| 正常传参场景  | 转换 .cu 文件并产出 HipifyResult     | 已覆盖（basic_cu_conversion、empty_project_dir）      |
| 异常传参场景  | 目录不存在 SystemExit 与缺参异常     | 已覆盖（nonexistent_dir_exits、missing_arg_raises）   |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合        | 未覆盖（纯源码目录批量转换入口，无 Tensor 输入）      |

未覆盖项及原因：
- 枚举选项：hipify 可选开关众多，本补充用例聚焦核心转换、空目录与异常路径，未穷举全部布尔标志组合。
- 混合设备类型：接口为源码目录级 CUDA-to-HIP 转换工具，无 Tensor 输入，不涉及设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import contextlib
import io
import os
import tempfile
import unittest

import torch
from torch.utils.hipify import hipify_python
from torch.utils.hipify.hipify_python import CurrentState, HipifyResult, hipify

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
class TestHipifyFunction(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.hipify."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self._saved_final_result = hipify_python.HIPIFY_FINAL_RESULT.copy()
        hipify_python.HIPIFY_FINAL_RESULT.clear()

    def tearDown(self):
        hipify_python.HIPIFY_FINAL_RESULT.clear()
        hipify_python.HIPIFY_FINAL_RESULT.update(self._saved_final_result)
        super().tearDown()

    def test_basic_cu_conversion(self):
        """Converts a .cu file into a hipified output file with DONE state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "src")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(src_dir)
            cu_path = os.path.join(src_dir, "sample.cu")
            with open(cu_path, "w", encoding="utf-8") as f:
                f.write("void alloc(void** p) { cudaMalloc(p, 16); }\n")
            with contextlib.redirect_stderr(io.StringIO()):
                results = hipify(project_directory=src_dir, output_directory=out_dir, show_progress=False)
            self.assertIsInstance(results, dict)
            self.assertEqual(len(results), 1)
            res = next(iter(results.values()))
            self.assertIsInstance(res, HipifyResult)
            self.assertEqual(res.current_state, CurrentState.DONE)
            self.assertTrue(os.path.exists(res.hipified_path))
            with open(res.hipified_path, "r", encoding="utf-8") as f:
                hip_content = f.read()
            self.assertIn("hipMalloc", hip_content)

    def test_empty_project_dir(self):
        """Empty project directory returns an empty result mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "empty_src")
            out_dir = os.path.join(tmpdir, "empty_out")
            os.makedirs(src_dir)
            with contextlib.redirect_stderr(io.StringIO()):
                results = hipify(project_directory=src_dir, output_directory=out_dir, show_progress=False)
            self.assertEqual(results, {})

    def test_nonexistent_dir_exits(self):
        """Nonexistent project_directory terminates with SystemExit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_dir = os.path.join(tmpdir, "no_such_dir")
            out_buf = io.StringIO()
            with contextlib.redirect_stdout(out_buf):
                with self.assertRaises(SystemExit):
                    hipify(project_directory=missing_dir, show_progress=False)

    def test_missing_arg_raises(self):
        """Calling hipify without project_directory raises TypeError."""
        with self.assertRaises(TypeError):
            hipify()


if __name__ == "__main__":
    run_tests()
