# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.datapipes.iter.streamreader.StreamReaderIterDataPipe 接口功能正确性
API 名称：torch.utils.data.datapipes.iter.streamreader.StreamReaderIterDataPipe
API 签名：StreamReaderIterDataPipe(datapipe, chunk=None)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 空字节流与非空字节流读取                   | 已覆盖（empty_stream、read_all_default）               |
| 枚举选项         | 接口枚举类参数覆盖                         | 未覆盖（接口无枚举类参数）                             |
| 参数类型         | IterDataPipe 与 chunk（None/int）入参      | 已覆盖（read_all_default、read_in_chunks）             |
| 传参与不传参     | 省略 chunk 默认整段读取与显式传入 chunk    | 已覆盖（read_all_default、read_in_chunks）             |
| 等价类/边界值    | 非整除分块、多流串联与空流边界             | 已覆盖（uneven_chunk_and_multi_stream、empty_stream）  |
| 正常传参场景     | 整段读取、定长分块读取与函数式别名调用     | 已覆盖（read_all_default、read_in_chunks、functional_read_from_stream） |
| 异常传参场景     | 缺少必填参数 datapipe                      | 已覆盖（missing_arg_raises）                           |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（纯字节流读取 DataPipe，不接收 Tensor 输入）    |

未覆盖项及原因：
- 枚举选项：接口无枚举类参数。
- 混合设备类型：纯二进制流分块读取 DataPipe，不涉及 Tensor 与设备迁移。

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
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.iter.streamreader import StreamReaderIterDataPipe

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
class TestStreamReaderIterDataPipe(TestCase):
    """Test cases for StreamReaderIterDataPipe."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_read_all_default(self):
        """With default chunk=None, reads entire stream as a single chunk."""
        buf = io.BytesIO(b"abcdef")
        dp = IterableWrapper([("f1", buf)])
        reader = StreamReaderIterDataPipe(dp)
        result = list(reader)
        self.assertEqual(result, [("f1", b"abcdef")])
        self.assertEqual(len(result), 1)

    def test_read_in_chunks(self):
        """With chunk=2, yields fixed-size byte chunks paired with stream label."""
        buf = io.BytesIO(b"abcdef")
        dp = IterableWrapper([("f1", buf)])
        reader = StreamReaderIterDataPipe(dp, chunk=2)
        result = list(reader)
        self.assertEqual(result, [("f1", b"ab"), ("f1", b"cd"), ("f1", b"ef")])
        self.assertEqual(b"".join(chunk for _, chunk in result), b"abcdef")

    def test_uneven_chunk_and_multi_stream(self):
        """Handles non-divisible chunk sizes and multiple consecutive streams."""
        buf1 = io.BytesIO(b"abcde")
        buf2 = io.BytesIO(b"123")
        dp = IterableWrapper([("s1", buf1), ("s2", buf2)])
        reader = StreamReaderIterDataPipe(dp, chunk=2)
        result = list(reader)
        self.assertEqual(
            result,
            [("s1", b"ab"), ("s1", b"cd"), ("s1", b"e"), ("s2", b"12"), ("s2", b"3")],
        )
        self.assertEqual(len(result), 5)

    def test_empty_stream(self):
        """Empty BytesIO yields no items cleanly."""
        buf = io.BytesIO(b"")
        dp = IterableWrapper([("empty", buf)])
        reader = StreamReaderIterDataPipe(dp, chunk=4)
        self.assertEqual(list(reader), [])
        self.assertEqual(buf.getvalue(), b"")

    def test_functional_read_from_stream(self):
        """Functional alias read_from_stream constructs StreamReaderIterDataPipe."""
        buf = io.BytesIO(b"xyz")
        dp = IterableWrapper([("fn", buf)]).read_from_stream(chunk=2)
        self.assertIsInstance(dp, StreamReaderIterDataPipe)
        self.assertEqual(list(dp), [("fn", b"xy"), ("fn", b"z")])

    def test_missing_arg_raises(self):
        """Constructing StreamReaderIterDataPipe without datapipe raises TypeError."""
        with self.assertRaises(TypeError):
            StreamReaderIterDataPipe()


if __name__ == "__main__":
    run_tests()
