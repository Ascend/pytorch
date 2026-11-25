# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.backend_registration.generate_methods_for_privateuse1_backend 接口功能正确性
API 名称：torch.utils.backend_registration.generate_methods_for_privateuse1_backend
API 签名：generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True,
         for_storage=False, for_packed_sequence=True, unsupported_dtype=None)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | unsupported_dtype 默认 None 与非空   | 已覆盖（default_flags_after_rename、storage_and_unsupported_dtype） |
| 枚举选项         | 各布尔注册开关组合                   | 已覆盖（default_flags_after_rename、storage_and_unsupported_dtype） |
| 参数类型         | 布尔开关与 dtype 列表参数类型        | 已覆盖（storage_and_unsupported_dtype）            |
| 传参与不传参     | 全默认无参调用与显式关键字传参       | 已覆盖（default_flags_after_rename、storage_and_unsupported_dtype） |
| 等价类/边界值    | 重复注册与禁用 dtype 拦截边界        | 已覆盖（duplicate_registration_raises_runtime_error 等） |
| 正常传参场景     | 子进程隔离验证 Tensor/Module/Storage | 已覆盖（default_flags_after_rename、storage_and_unsupported_dtype） |
| 异常传参场景     | 重复注册 RuntimeError 与非法关键字   | 已覆盖（duplicate_registration_raises_runtime_error、unexpected_kwarg_raises） |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 混合设备类型：后端方法注册接口不接收多 Tensor 输入，不涉及混合设备场景。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import subprocess
import sys
import textwrap
import unittest

import torch
from torch.utils.backend_registration import generate_methods_for_privateuse1_backend

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
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGenerateMethodsForPrivateUse1Backend(TestCase):
    """Test cases for generate_methods_for_privateuse1_backend."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def _run_clean_subprocess(self, script):
        env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
        proc = subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                              env=env, capture_output=True, text=True, check=False)
        msg = f"Subprocess failed (rc={proc.returncode}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        self.assertEqual(proc.returncode, 0, msg)

    def test_default_flags_after_rename(self):
        """Default flags generate backend methods on Tensor, Module, and PackedSequence only."""
        self._run_clean_subprocess("""
            import torch
            import torch.utils.backend_registration as br
            br.rename_privateuse1_backend("mydev")
            assert br.generate_methods_for_privateuse1_backend() is None
            assert hasattr(torch.Tensor, "mydev") and hasattr(torch.Tensor, "is_mydev")
            assert hasattr(torch.nn.Module, "mydev") and hasattr(torch.nn.utils.rnn.PackedSequence, "mydev")
            assert not hasattr(torch.storage._StorageBase, "mydev")
            assert torch.tensor([1.0]).is_mydev is False
        """)

    def test_storage_and_unsupported_dtype(self):
        """Enabling for_storage with unsupported_dtype generates storage methods and blocks dtype."""
        self._run_clean_subprocess("""
            import warnings, torch
            import torch.utils.backend_registration as br
            br.rename_privateuse1_backend("mydev")
            br.generate_methods_for_privateuse1_backend(for_tensor=False, for_module=False,
                for_packed_sequence=False, for_storage=True, unsupported_dtype=[torch.float64])
            assert not hasattr(torch.Tensor, "mydev") and not hasattr(torch.nn.Module, "mydev")
            assert hasattr(torch.storage._StorageBase, "mydev") and hasattr(torch.TypedStorage, "mydev")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    torch.TypedStorage(2, dtype=torch.float64).mydev()
                    raise AssertionError("Expected RuntimeError for unsupported dtype")
                except RuntimeError as exc:
                    assert "float64" in str(exc) or "not supported" in str(exc)
        """)

    def test_duplicate_registration_raises_runtime_error(self):
        """Calling generate_methods_for_privateuse1_backend twice raises RuntimeError."""
        self._run_clean_subprocess("""
            import torch.utils.backend_registration as br
            br.rename_privateuse1_backend("mydev")
            br.generate_methods_for_privateuse1_backend()
            try:
                br.generate_methods_for_privateuse1_backend()
                raise AssertionError("Expected RuntimeError on duplicate registration")
            except RuntimeError:
                pass
        """)

    def test_unexpected_kwarg_raises(self):
        """Passing an unexpected keyword argument raises TypeError."""
        with self.assertRaises(TypeError):
            generate_methods_for_privateuse1_backend(unknown_arg=True)


if __name__ == "__main__":
    run_tests()
