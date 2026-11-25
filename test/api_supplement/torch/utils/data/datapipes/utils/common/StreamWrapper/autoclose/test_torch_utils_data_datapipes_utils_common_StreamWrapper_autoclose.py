# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.datapipes.utils.common.StreamWrapper.autoclose 接口功能正确性
API 名称：torch.utils.data.datapipes.utils.common.StreamWrapper.autoclose
API 签名：StreamWrapper.autoclose(self)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 无子流与含单/多子流的关闭时机              | 已覆盖（autoclose_no_children、autoclose_with_child）  |
| 枚举选项         | 接口枚举类参数覆盖                         | 未覆盖（接口无枚举类参数）                             |
| 参数类型         | 不同入参类型覆盖                           | 未覆盖（无参实例方法，仅接收 self）                    |
| 传参与不传参     | 可选默认参数省略与显式传入                 | 未覆盖（无可选参数，不接受额外入参）                   |
| 等价类/边界值    | 无子流立即关闭、末子流关闭触发与幂等重复调 | 已覆盖（autoclose_no_children、autoclose_multiple_children、autoclose_idempotent） |
| 正常传参场景     | 正常触发 autoclose 并校验关闭状态          | 已覆盖（autoclose_no_children、autoclose_with_child、autoclose_idempotent） |
| 异常传参场景     | 传入多余位置参数触发 TypeError             | 已覆盖（unexpected_arg_raises）                        |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（纯主机 IO 流生命周期管理，不接收 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口无枚举类参数。
- 参数类型：autoclose 为无参实例方法，不接收除 self 外的入参。
- 传参与不传参：无可选默认参数。
- 混合设备类型：纯主机 IO 流关闭控制方法，不涉及 Tensor 与设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import io
import os
import unittest

import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper

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
class TestStreamWrapperAutoclose(TestCase):
    """Test cases for torch.utils.data.datapipes.utils.common.StreamWrapper.autoclose."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_autoclose_no_children(self):
        """Closes stream immediately when autoclose is called with no child streams."""
        buf = io.BytesIO(b"abc")
        sw = StreamWrapper(buf)
        self.assertFalse(sw.close_on_last_child)
        self.assertFalse(sw.closed)
        ret = sw.autoclose()
        self.assertIsNone(ret)
        self.assertTrue(sw.close_on_last_child)
        self.assertTrue(sw.closed)
        self.assertTrue(buf.closed)

    def test_autoclose_with_child(self):
        """Defers closing parent stream until its child stream is closed."""
        parent_buf = io.BytesIO(b"parent")
        child_buf = io.BytesIO(b"child")
        parent = StreamWrapper(parent_buf)
        child = StreamWrapper(child_buf, parent_stream=parent)
        ret = parent.autoclose()
        self.assertIsNone(ret)
        self.assertTrue(parent.close_on_last_child)
        self.assertFalse(parent.closed)
        self.assertFalse(parent_buf.closed)
        child.close()
        self.assertTrue(child.closed)
        self.assertTrue(parent.closed)
        self.assertTrue(parent_buf.closed)

    def test_autoclose_multiple_children(self):
        """Keeps parent open until the last remaining child stream closes."""
        parent_buf = io.BytesIO(b"parent")
        parent = StreamWrapper(parent_buf)
        c1 = StreamWrapper(io.BytesIO(b"c1"), parent_stream=parent)
        c2 = StreamWrapper(io.BytesIO(b"c2"), parent_stream=parent)
        parent.autoclose()
        c1.close()
        self.assertFalse(parent.closed)
        c2.close()
        self.assertTrue(parent.closed)
        self.assertTrue(parent_buf.closed)

    def test_autoclose_idempotent(self):
        """Calling autoclose repeatedly on a closed stream is a safe no-op."""
        buf = io.BytesIO(b"data")
        sw = StreamWrapper(buf)
        sw.autoclose()
        self.assertIsNone(sw.autoclose())
        self.assertTrue(sw.closed)
        self.assertTrue(buf.closed)

    def test_unexpected_arg_raises(self):
        """Passing an unexpected argument to autoclose raises TypeError."""
        sw = StreamWrapper(io.BytesIO(b"abc"))
        try:
            with self.assertRaises(TypeError):
                sw.autoclose(True)
        finally:
            sw.close()


if __name__ == "__main__":
    run_tests()
