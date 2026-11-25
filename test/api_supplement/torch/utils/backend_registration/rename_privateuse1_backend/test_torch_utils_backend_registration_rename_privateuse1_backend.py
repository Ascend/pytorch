# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.backend_registration.rename_privateuse1_backend 接口功能正确性
API 名称：torch.utils.backend_registration.rename_privateuse1_backend
API 签名：torch.utils.backend_registration.rename_privateuse1_backend(backend_name)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 非空后端名字符串与空字符串边界       | 未覆盖（见下）                                     |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（见下）                                     |
| 参数类型         | 合法字符串与非法整型参数             | 已覆盖（invalid_type_and_missing_arg_raise_type_error） |
| 传参与不传参     | 传入必选参数与缺少参数               | 已覆盖（invalid_type_and_missing_arg_raise_type_error） |
| 等价类/边界值    | 首次重命名与重复重命名边界           | 已覆盖（rename_twice_raises_runtime_error 等）     |
| 正常传参场景     | 子进程内重命名并验证设备解析         | 已覆盖（rename_sets_backend_name_and_device_type） |
| 异常传参场景     | 重复重命名、非法类型及缺参报错       | 已覆盖（rename_twice_raises_runtime_error 等）     |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 空/非空：未单独构造空字符串后端名用例（重命名全局不可逆，仅测合法非空名与非法类型）。
- 枚举选项：接口仅接收自定义字符串名称，无枚举参数。
- 混合设备类型：全局后端名称注册接口，不接收 Tensor 输入。

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
from torch.utils.backend_registration import rename_privateuse1_backend

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
class TestRenamePrivateUse1Backend(TestCase):
    """Test cases for torch.utils.backend_registration.rename_privateuse1_backend."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _run_clean_subprocess(self, script):
        env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
        cmd = [sys.executable, "-c", textwrap.dedent(script)]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        msg = f"Subprocess failed (rc={proc.returncode}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        self.assertEqual(proc.returncode, 0, msg)

    def test_rename_sets_backend_name_and_device_type(self):
        """Renaming privateuse1 updates backend name and enables torch.device parsing."""
        self._run_clean_subprocess("""
            import torch
            from torch.utils.backend_registration import rename_privateuse1_backend
            assert torch._C._get_privateuse1_backend_name() == "privateuseone"
            assert rename_privateuse1_backend("mydev") is None
            assert torch._C._get_privateuse1_backend_name() == "mydev"
            assert torch.device("mydev").type == "mydev"
            assert torch.device("mydev:0").index == 0
        """)

    def test_rename_twice_raises_runtime_error(self):
        """Calling rename_privateuse1_backend a second time raises RuntimeError."""
        self._run_clean_subprocess("""
            import torch
            from torch.utils.backend_registration import rename_privateuse1_backend
            rename_privateuse1_backend("mydev")
            try:
                rename_privateuse1_backend("otherdev")
                raise AssertionError("Expected RuntimeError on second rename")
            except RuntimeError:
                pass
            assert torch._C._get_privateuse1_backend_name() == "mydev"
        """)

    def test_invalid_type_and_missing_arg_raise_type_error(self):
        """Passing a non-string argument or omitting backend_name raises an error."""
        self._run_clean_subprocess("""
            import torch
            from torch.utils.backend_registration import rename_privateuse1_backend
            for call in (lambda: rename_privateuse1_backend(123), lambda: rename_privateuse1_backend()):
                try:
                    call()
                    raise AssertionError("Expected TypeError")
                except (TypeError, RuntimeError):
                    pass
            assert torch._C._get_privateuse1_backend_name() == "privateuseone"
        """)

    def test_in_process_already_renamed_raises_runtime_error(self):
        """In-process rename raises RuntimeError when already renamed or TypeError if missing arg."""
        current_backend = torch._C._get_privateuse1_backend_name()
        if current_backend != "privateuseone":
            with self.assertRaises(RuntimeError):
                rename_privateuse1_backend("another_backend")
            self.assertEqual(torch._C._get_privateuse1_backend_name(), current_backend)
        else:
            with self.assertRaises(TypeError):
                rename_privateuse1_backend()


if __name__ == "__main__":
    run_tests()
