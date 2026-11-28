# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.compiler.PrecompileError 接口功能正确性
API 名称：torch.compiler.PrecompileError
API 签名：PrecompileError(*args)

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 零实参、None、空/非空消息及容器 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | bool/int/float/str/list/tuple/dict 消息、多个实参和缺失产物 Path |
| 传参与不传参 | 不传消息、单个消息和多个位置实参 |
| 等价类/边界值 | 零/负值、空消息、pickle 身份及真实产物读取失败的异常链 |
| 正常传参场景 | 公开异常类型、消息与 pickle 身份 |
| 异常传参场景 | 未知关键字 TypeError；产物缺失的 PrecompileError/OSError cause |
| 混合设备类型 | 不适用：本文件为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
DEVICE_NAME = None
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch_npu":
            raise
    else:
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    if DEVICE_NAME is None and torch.cuda.is_available():
        DEVICE_NAME = "cuda"

if DEVICE_NAME == "npu":
    from torch_npu.testing.testcase import TestCase, run_tests
else:
    from unittest import TestCase

    def run_tests():
        unittest.main()

from pathlib import Path
import pickle
import tempfile
from torch.compiler import precompile as pc


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_precompileerror_error_is_runtime_error_and_pickle_retains_public_identity(self):
        error = torch.compiler.PrecompileError("controlled precompile error")
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(error.args, ("controlled precompile error",))
        restored = pickle.loads(pickle.dumps(error))
        self.assertIs(type(restored), torch.compiler.PrecompileError)
        self.assertEqual(str(restored), "controlled precompile error")
        self.assertIs(pc.PrecompileError, torch.compiler.PrecompileError)

    def test_precompileerror_actual_missing_artifact_raises_public_error_with_file_cause(self):
        with tempfile.TemporaryDirectory(prefix="graph-ut-error-") as directory:
            with self.assertRaises(torch.compiler.PrecompileError) as raised:
                pc.load(Path(directory) / "absent.py", Path(directory) / "absent.cache")
            self.assertIsInstance(raised.exception.__cause__, OSError)


    def test_parameter_empty_single_and_multiple_exception_arguments(self):
        empty = torch.compiler.PrecompileError()
        self.assertEqual(empty.args, ())
        self.assertEqual(str(empty), "")
        for payload in (None, False, 0, -1, 2.5, "", "message", [], [1], (), (1,), {}, {"x": 2}):
            with self.subTest(payload=payload):
                error = torch.compiler.PrecompileError(payload)
                self.assertEqual(error.args, (payload,))
                self.assertEqual(str(error), str(payload))
        error = torch.compiler.PrecompileError("message", None, 3)
        self.assertEqual(error.args, ("message", None, 3))
        self.assertEqual(str(error), str(("message", None, 3)))
        with self.assertRaises(TypeError):
            torch.compiler.PrecompileError(message="unsupported keyword")


if __name__ == "__main__":
    run_tests()
