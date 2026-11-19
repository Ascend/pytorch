# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.utils.rnn.invert_permutation 接口功能正确性
API 名称：torch.nn.utils.rnn.invert_permutation
API 签名：invert_permutation(permutation: Tensor | None) -> Tensor | None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | permutation=None 返回 None；空张量返回同 dtype 空张量 | 已覆盖（test_invert_permutation_none、
test_invert_permutation_boundary_empty） |
| 枚举选项 | 已知值、随机排列、非连续视图、单元素、大张量 | 已覆盖（test_invert_permutation_basic_known_values、
test_invert_permutation_inverse_property_random、test_invert_permutation_non_contiguous_input、
test_invert_permutation_boundary_single_element、test_invert_permutation_boundary_large_perm） |
| 参数类型 | int64 输入返回 int64、设备一致；int32 输入报 RuntimeError | 已覆盖（test_invert_permutation_dtype_device_preserved、
test_invert_permutation_cpu_int32_raises） |
| 传参与不传参 | 显式 Tensor 与 None 两种传参 | 已覆盖（test_invert_permutation_none、
test_invert_permutation_basic_known_values） |
| 等价类/边界值 | 空张量、单元素、1000 元素排列、非连续输入 | 已覆盖（test_invert_permutation_boundary_empty、
test_invert_permutation_boundary_single_element、test_invert_permutation_boundary_large_perm、
test_invert_permutation_non_contiguous_input） |
| 正常传参场景 | 数学契约 inv[perm]==arange(n) 与 invert(invert(p))==p；设备结果与
CPU Golden 精确一致；重复执行确定 | 已覆盖（test_invert_permutation_basic_known_values、
test_invert_permutation_double_inverse_identity、test_invert_permutation_inverse_property_random、
test_invert_permutation_deterministic_fixed_seed） |
| 异常传参场景 | int32 dtype、2D 输入、越界索引在 CPU 输入上触发稳定原生报错 | 已覆盖（test_invert_permutation_cpu_int32_raises、
test_invert_permutation_cpu_2d_raises、test_invert_permutation_cpu_out_of_bounds_raises） |
| 混合设备类型 | CPU 排列移至设备后 invert 与 CPU Golden 逐元素精确一致 | 已覆盖（test_invert_permutation_basic_known_values、
test_invert_permutation_inverse_property_random） |

未覆盖项及原因：
- 越界/重复索引在设备端的行为未定义（设备断言或静默写坏），仅以 CPU 输入看护原生报错。
- 该 API 无浮点输出、无精度风险，小值域维度不适用。

注意：整数结果使用精确断言（torch.testing.assert_close 对整数 dtype 默认 rtol=0, atol=0）；
     设备输出移至 CPU 后与 CPU Golden 对照；CPU 专属异常契约以 CPU 张量固定执行，不依赖设备。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化；
# 被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
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

import torch.nn.utils.rnn


def _assert_inverse(perm):
    """Assert the mathematical inverse contract inv[perm] == arange(n) exactly."""
    inv = torch.nn.utils.rnn.invert_permutation(perm)
    arange = torch.arange(perm.numel(), device=perm.device)
    torch.testing.assert_close(inv[perm].cpu(), arange.cpu(), rtol=0.0, atol=0.0)
    return inv


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestInvertPermutation(TestCase):
    """Test cases for torch.nn.utils.rnn.invert_permutation (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_invert_permutation_basic_known_values(self):
        """Known permutation inverts exactly and matches the CPU Golden."""
        perm_cpu = torch.tensor([2, 0, 3, 1])
        golden = torch.nn.utils.rnn.invert_permutation(perm_cpu)
        self.assertEqual(golden.tolist(), [1, 3, 0, 2])
        perm_device = perm_cpu.to(self.device_name)
        inv = torch.nn.utils.rnn.invert_permutation(perm_device)
        self.assertEqual(inv.device.type, self.device_name)
        torch.testing.assert_close(inv.cpu(), golden, rtol=0.0, atol=0.0)

    def test_invert_permutation_none(self):
        """invert_permutation(None) returns None for both call orders."""
        self.assertIsNone(torch.nn.utils.rnn.invert_permutation(None))
        self.assertIsNone(
            torch.nn.utils.rnn.invert_permutation(
                torch.nn.utils.rnn.invert_permutation(None)
            )
        )

    def test_invert_permutation_boundary_empty(self):
        """An empty int64 tensor yields an empty int64 result on the same device."""
        empty = torch.tensor([], dtype=torch.int64, device=self.device_name)
        inv = torch.nn.utils.rnn.invert_permutation(empty)
        self.assertEqual(tuple(inv.shape), (0,))
        self.assertEqual(inv.dtype, torch.int64)
        self.assertEqual(inv.device.type, self.device_name)

    def test_invert_permutation_boundary_single_element(self):
        """A single-element permutation [0] inverts to itself."""
        perm = torch.tensor([0], device=self.device_name)
        inv = _assert_inverse(perm)
        self.assertEqual(inv.cpu().tolist(), [0])

    def test_invert_permutation_double_inverse_identity(self):
        """invert is an involution: invert(invert(p)) == p exactly on device."""
        perm = torch.tensor([3, 1, 0, 2], device=self.device_name)
        inv = torch.nn.utils.rnn.invert_permutation(perm)
        twice = torch.nn.utils.rnn.invert_permutation(inv)
        torch.testing.assert_close(twice.cpu(), perm.cpu(), rtol=0.0, atol=0.0)
        self.assertIsNone(
            torch.nn.utils.rnn.invert_permutation(
                torch.nn.utils.rnn.invert_permutation(None)
            )
        )

    def test_invert_permutation_inverse_property_random(self):
        """Random permutations satisfy inv[perm] == arange(n) exactly on device."""
        torch.manual_seed(42)
        for size in [4, 16, 64]:
            perm = torch.randperm(size).to(self.device_name)
            _assert_inverse(perm)

    def test_invert_permutation_boundary_large_perm(self):
        """A 1000-element permutation satisfies the inverse contract on device."""
        torch.manual_seed(123)
        perm = torch.randperm(1000).to(self.device_name)
        inv = _assert_inverse(perm)
        self.assertEqual(inv.numel(), 1000)

    def test_invert_permutation_dtype_device_preserved(self):
        """Output dtype is int64 and output device matches the input device."""
        perm = torch.tensor([1, 0], dtype=torch.int64, device=self.device_name)
        inv = torch.nn.utils.rnn.invert_permutation(perm)
        self.assertEqual(inv.dtype, torch.int64)
        self.assertEqual(inv.device.type, perm.device.type)

    def test_invert_permutation_non_contiguous_input(self):
        """A strided view input still yields the exact inverse of its values."""
        torch.manual_seed(5)
        buf = torch.zeros(8, dtype=torch.int64)
        buf[::2] = torch.randperm(4)
        perm_cpu = buf[::2]
        self.assertEqual(perm_cpu.numel(), 4)
        self.assertEqual(sorted(perm_cpu.tolist()), [0, 1, 2, 3])
        self.assertFalse(perm_cpu.is_contiguous())
        golden = torch.nn.utils.rnn.invert_permutation(perm_cpu)
        inv = torch.nn.utils.rnn.invert_permutation(perm_cpu.to(self.device_name))
        torch.testing.assert_close(inv.cpu(), golden, rtol=0.0, atol=0.0)

    def test_invert_permutation_cpu_int32_raises(self):
        """int32 input hits the scatter dtype check and raises RuntimeError (CPU contract)."""
        perm = torch.tensor([2, 0, 1], dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            torch.nn.utils.rnn.invert_permutation(perm)

    def test_invert_permutation_cpu_2d_raises(self):
        """2-D input raises IndexError because scatter runs along dim 0 only (CPU contract)."""
        perm = torch.tensor([[2, 0], [1, 3]])
        with self.assertRaises(IndexError):
            torch.nn.utils.rnn.invert_permutation(perm)

    def test_invert_permutation_cpu_out_of_bounds_raises(self):
        """Out-of-bounds indices raise RuntimeError on CPU (CPU contract)."""
        perm = torch.tensor([0, 5])
        with self.assertRaises(RuntimeError):
            torch.nn.utils.rnn.invert_permutation(perm)

    def test_invert_permutation_deterministic_fixed_seed(self):
        """Fixed seed fixes the permutation; repeated device runs match bit-exactly."""
        torch.manual_seed(99)
        perm_cpu = torch.randperm(32)
        perm = perm_cpu.to(self.device_name)
        first = torch.nn.utils.rnn.invert_permutation(perm)
        second = torch.nn.utils.rnn.invert_permutation(perm)
        torch.testing.assert_close(first.cpu(), second.cpu(), rtol=0.0, atol=0.0)

    def test_invert_permutation_memory_repeated_calls(self):
        """50 repeated device calls do not accumulate device memory."""
        if self.device_name == 'cpu':
            self.skipTest("requires a device allocator (NPU/CUDA)")
        perm = torch.randperm(64).to(self.device_name)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        for _ in range(50):
            inv = torch.nn.utils.rnn.invert_permutation(perm)
            del inv
        torch.cuda.empty_cache()
        grown = torch.cuda.memory_allocated()
        self.assertLessEqual(grown - base, 1024 * 1024)


if __name__ == "__main__":
    run_tests()
