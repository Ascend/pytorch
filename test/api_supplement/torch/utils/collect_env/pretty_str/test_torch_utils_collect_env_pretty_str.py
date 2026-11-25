# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.collect_env.pretty_str 接口功能正确性
API 名称：torch.utils.collect_env.pretty_str
API 签名：torch.utils.collect_env.pretty_str(envinfo)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | None/空字符串字段与非空字段渲染      | 已覆盖（none_and_empty_replacements、basic_format） |
| 枚举选项         | 布尔字段 True/False 转 Yes/No        | 已覆盖（bool_replacements）                        |
| 参数类型         | SystemEnv 实例与非法 dict 参数       | 已覆盖（basic_format、missing_arg_and_invalid_type_raises） |
| 传参与不传参     | 必选 envinfo 传入与缺参              | 已覆盖（basic_format、missing_arg_and_invalid_type_raises） |
| 等价类/边界值    | None/空串占位替换与多行 GPU 文本     | 已覆盖（none_and_empty_replacements、multiline_gpu_and_packages） |
| 正常传参场景     | 基本格式、布尔、空值与多行格式化     | 已覆盖（basic_format、bool_replacements、multiline_gpu_and_packages 等） |
| 异常传参场景     | 缺参 TypeError 与非法 dict 报 AttributeError | 已覆盖（missing_arg_and_invalid_type_raises） |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 混合设备类型：纯 Python 环境信息字符串格式化函数，不接收 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.collect_env import SystemEnv, pretty_str

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


def _make_env(**overrides):
    fields = dict.fromkeys(SystemEnv._fields)
    fields.update({
        "torch_version": "2.15.0",
        "is_debug_build": False,
        "os": "Ubuntu 22.04 LTS",
        "python_version": "3.11.0",
        "python_platform": "Linux-x86_64",
        "is_cuda_available": False,
        "is_xpu_available": False,
        "pip_version": "pip3",
        "pip_packages": "torch==2.15.0",
        "conda_packages": "",
        "caching_allocator_config": "",
        "is_xnnpack_available": "True",
        "cpu_info": "x86_64",
    })
    fields.update(overrides)
    return SystemEnv(**fields)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestPrettyStr(TestCase):
    """Test cases for torch.utils.collect_env.pretty_str."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_basic_format(self):
        """Standard SystemEnv fields are formatted into the expected summary lines."""
        text = pretty_str(_make_env())
        self.assertIsInstance(text, str)
        self.assertIn("PyTorch version: 2.15.0", text)
        self.assertIn("OS: Ubuntu 22.04 LTS", text)
        self.assertIn("Python version: 3.11.0", text)
        self.assertIn("[pip3] torch==2.15.0", text)

    def test_bool_replacements(self):
        """Boolean flags render as 'Yes' and 'No' in the formatted output."""
        text = pretty_str(_make_env(is_debug_build=True, is_cuda_available=False))
        self.assertIn("Is debug build: Yes", text)
        self.assertIn("Is CUDA available: No", text)

    def test_none_and_empty_replacements(self):
        """None and empty fields render fallback placeholders accurately."""
        text = pretty_str(_make_env(gcc_version=None, pip_packages="", caching_allocator_config=""))
        self.assertIn("GCC version: Could not collect", text)
        self.assertIn("[pip3] No relevant packages", text)
        self.assertIn("Caching allocator config: N/A", text)

    def test_multiline_gpu_and_packages(self):
        """Multiline GPU model descriptions are formatted on subsequent lines."""
        text = pretty_str(_make_env(nvidia_gpu_models="GPU 0: A\nGPU 1: B"))
        self.assertIn("GPU models and configuration: \nGPU 0: A\nGPU 1: B\n", text)

    def test_missing_arg_and_invalid_type_raises(self):
        """Missing argument or passing a plain dict raises TypeError or AttributeError."""
        with self.assertRaises(TypeError):
            pretty_str()
        with self.assertRaises(AttributeError):
            pretty_str({"torch_version": "2.15.0"})


if __name__ == "__main__":
    run_tests()
