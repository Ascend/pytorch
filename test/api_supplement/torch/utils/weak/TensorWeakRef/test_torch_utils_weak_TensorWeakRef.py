# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.weak.TensorWeakRef 接口功能正确性
API 名称：torch.utils.weak.TensorWeakRef
API 签名：torch.utils.weak.TensorWeakRef(tensor: Tensor)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空 Tensor（0 元素）与非空 Tensor     | 已覆盖（empty_tensor、                             |
|                  |                                      | cpu_tensor_live_and_collected）                    |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（无枚举类参数）                             |
| 参数类型         | CPU/目标设备 Tensor 与非 Tensor 对象 | 已覆盖（cpu_tensor_live_and_collected、            |
|                  |                                      | device_tensor_live_and_collected、                 |
|                  |                                      | non_tensor_raises）                                |
| 传参与不传参     | 必填 tensor 参数传入与缺参           | 已覆盖（cpu_tensor_live_and_collected、            |
|                  |                                      | missing_arg_raises）                               |
| 等价类/边界值    | 存活解引用、GC 回收后返回 None 与    | 已覆盖（cpu_tensor_live_and_collected、            |
|                  | 多弱引用指向同一 Tensor              | device_tensor_live_and_collected、                 |
|                  |                                      | multiple_refs_same_tensor）                        |
| 正常传参场景     | CPU 与设备侧 Tensor 弱引用生命周期   | 已覆盖（cpu_tensor_live_and_collected、            |
|                  |                                      | device_tensor_live_and_collected、empty_tensor）   |
| 异常传参场景     | 非 Tensor 参数与缺参异常             | 已覆盖（non_tensor_raises、missing_arg_raises）    |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（单 Tensor 弱引用封装，无多 Tensor 输入）   |

未覆盖项及原因：
- 枚举选项：该接口仅接收单个 Tensor 参数，无枚举类参数。
- 混合设备类型：该接口每次仅封装单个 Tensor 的弱引用，不涉及多 Tensor 混合设备运算。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import gc
import os
import unittest

import torch
from torch.utils.weak import TensorWeakRef

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
class TestTensorWeakRef(TestCase):
    """Test cases for torch.utils.weak.TensorWeakRef."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_cpu_tensor_live_and_collected(self):
        """Dereferences a live CPU tensor and returns None after garbage collection."""
        expected = torch.tensor([1.0, 2.5, -3.0])
        tensor = expected.clone()
        ref = TensorWeakRef(tensor)
        deref = ref()
        self.assertIs(deref, tensor)
        torch.testing.assert_close(deref, expected)
        del deref
        del tensor
        gc.collect()
        self.assertIsNone(ref())

    def test_device_tensor_live_and_collected(self):
        """Dereferences a live target-device tensor and returns None after GC."""
        expected = torch.tensor([4.0, -1.5, 2.0])
        tensor = expected.to(self.device)
        ref = TensorWeakRef(tensor)
        deref = ref()
        self.assertIs(deref, tensor)
        self.assertEqual(deref.device.type, self.device.type)
        torch.testing.assert_close(deref.cpu(), expected)
        del deref
        del tensor
        gc.collect()
        self.assertIsNone(ref())

    def test_empty_tensor(self):
        """Supports weak references to zero-element tensors until garbage collection."""
        tensor = torch.empty(0, device=self.device)
        ref = TensorWeakRef(tensor)
        self.assertIs(ref(), tensor)
        self.assertEqual(ref().numel(), 0)
        del tensor
        gc.collect()
        self.assertIsNone(ref())

    def test_multiple_refs_same_tensor(self):
        """Multiple TensorWeakRef instances to the same tensor expire together on GC."""
        tensor = torch.ones(2, 2, device=self.device)
        ref1 = TensorWeakRef(tensor)
        ref2 = TensorWeakRef(tensor)
        self.assertIs(ref1(), tensor)
        self.assertIs(ref2(), tensor)
        del tensor
        gc.collect()
        self.assertIsNone(ref1())
        self.assertIsNone(ref2())

    def test_non_tensor_raises(self):
        """Passing a non-Tensor object raises AssertionError or TypeError."""
        for bad in ([1, 2, 3], "not_a_tensor", None, 42):
            with self.assertRaises((AssertionError, TypeError)):
                TensorWeakRef(bad)

    def test_missing_arg_raises(self):
        """Calling TensorWeakRef without arguments raises TypeError."""
        with self.assertRaises(TypeError):
            TensorWeakRef()


if __name__ == "__main__":
    run_tests()
