# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.import_ir_module_from_buffer 接口功能正确性
API 名称：torch.import_ir_module_from_buffer
API 签名：torch.import_ir_module_from_buffer(cu, buffer, map_location, extra_files, load_debug_info) -> ScriptModule

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 传入已序列化模块字节流 | 已覆盖 |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | CompilationUnit/字节/设备映射/字典/布尔 | 已覆盖 |
| 传参与不传参 | 五个参数均为必填 | 已覆盖 |
| 等价类/边界值 | 保存-导入往返计算结果一致 | 已覆盖 |
| 正常传参场景 | 从内存字节流导入可执行模块 | 已覆盖 |
| 异常传参场景 | 非法字节流抛 RuntimeError | 已覆盖 |
| 混合设备类型 | 导入模块可在 CPU/加速器输入上执行 | 已覆盖 |

未覆盖项及原因：
- 本 API 无论文枚举参数，故该维度不适用。
- extra_files/load_debug_info 的调试元数据填充路径未逐项断言（仅传入默认值），
  属序列化辅助信息，不影响模块语义。
- 序列化产物使用 torch.jit.trace 生成（transfer_to_npu 下 torch.jit.script 被禁用），
  被测对象仍为 torch.import_ir_module_from_buffer。
"""
import io
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化，避免
# 不必要的探测开销；被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
DEVICE_NAME = 'cpu'
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = 'npu'
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

# ORDER CONSTRAINT: import torch_npu.testing.testcase BEFORE
# torch_npu.contrib.transfer_to_npu (patching first breaks the import).
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


class _ScaleModule(torch.nn.Module):
    """Minimal traceable module used to produce an in-memory IR buffer."""

    def forward(self, x):
        return x * 2.0


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class Testimport_ir_module_from_buffer(TestCase):
    """Test cases for torch.import_ir_module_from_buffer (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.input = torch.arange(4, dtype=torch.float32, device='cpu')
        self.module = torch.jit.trace(_ScaleModule(), self.input)
        self.expected = self.module(self.input)

    def _buffer_bytes(self):
        buf = io.BytesIO()
        torch.jit.save(self.module, buf)
        return buf.getvalue()

    def test_roundtrip_matches_traced_module(self):
        """Importing the in-memory IR reproduces the original module output."""
        cu = torch._C.CompilationUnit()
        loaded = torch.import_ir_module_from_buffer(cu, self._buffer_bytes(), None, {}, False)
        self.assertEqual(type(loaded).__name__, 'ScriptModule')
        torch.testing.assert_close(loaded.forward(self.input), self.expected)

    def test_loaded_module_runs_on_accelerator(self):
        """The imported module executes on the accelerator device identically."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accelerator device')
        cu = torch._C.CompilationUnit()
        loaded = torch.import_ir_module_from_buffer(cu, self._buffer_bytes(), None, {}, False)
        dev_input = self.input.to(self.device_name)
        torch.testing.assert_close(loaded.forward(dev_input).cpu(), self.expected)

    def test_garbage_buffer_raises(self):
        """An invalid buffer raises RuntimeError."""
        cu = torch._C.CompilationUnit()
        with self.assertRaises(RuntimeError):
            torch.import_ir_module_from_buffer(cu, b'not a torchscript archive', None, {}, False)


if __name__ == "__main__":
    run_tests()
