# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.datapipes.utils.common.StreamWrapper.close_streams 接口功能正确性
API 名称：torch.utils.data.datapipes.utils.common.StreamWrapper.close_streams
API 签名：StreamWrapper.close_streams(v, depth=0)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 空容器/非流对象与非空 StreamWrapper        | 已覆盖（non_stream_noop、close_single、close_containers） |
| 枚举选项         | 接口枚举类参数覆盖                         | 未覆盖（接口无枚举类参数）                             |
| 参数类型         | 单流、嵌套 list/tuple/dict 与非流标量      | 已覆盖（close_single、close_containers、non_stream_noop） |
| 传参与不传参     | 省略 depth 默认 0 与显式传入 depth         | 已覆盖（close_single、depth_limit_respected）          |
| 等价类/边界值    | depth 超过上限 10 提前返回与空容器无副作用 | 已覆盖（depth_limit_respected、non_stream_noop）       |
| 正常传参场景     | 关闭单流与递归关闭嵌套容器中的全部流       | 已覆盖（close_single、close_containers、non_stream_noop） |
| 异常传参场景     | 缺少必填参数 v                             | 已覆盖（missing_arg_raises）                           |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（纯主机 IO 流关闭工具，不接收 Tensor 输入）     |

未覆盖项及原因：
- 枚举选项：接口无枚举类参数。
- 混合设备类型：纯主机 IO 流递归关闭类方法，不涉及 Tensor 与设备迁移。

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
class TestStreamWrapperCloseStreams(TestCase):
    """Test cases for StreamWrapper.close_streams."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_close_single(self):
        """Closing a single StreamWrapper closes its underlying BytesIO."""
        buf = io.BytesIO(b"hello")
        sw = StreamWrapper(buf)
        ret = StreamWrapper.close_streams(sw)
        self.assertIsNone(ret)
        self.assertTrue(sw.closed)
        self.assertTrue(buf.closed)

    def test_close_containers(self):
        """Closing nested list, tuple, and dict closes every wrapped stream."""
        b1, b2, b3 = io.BytesIO(b"1"), io.BytesIO(b"2"), io.BytesIO(b"3")
        sw1, sw2, sw3 = StreamWrapper(b1), StreamWrapper(b2), StreamWrapper(b3)
        nested = {"list": [sw1, (sw2,)], "single": sw3, "plain": 42}
        ret = StreamWrapper.close_streams(nested)
        self.assertIsNone(ret)
        for sw, buf in ((sw1, b1), (sw2, b2), (sw3, b3)):
            self.assertTrue(sw.closed)
            self.assertTrue(buf.closed)

    def test_non_stream_noop(self):
        """Passing non-stream values or empty containers is a safe no-op."""
        for obj in (None, 123, "text", [], (), {}):
            self.assertIsNone(StreamWrapper.close_streams(obj))

    def test_depth_limit_respected(self):
        """When depth exceeds 10, close_streams returns early without closing."""
        buf = io.BytesIO(b"deep")
        sw = StreamWrapper(buf)
        try:
            ret = StreamWrapper.close_streams(sw, depth=11)
            self.assertIsNone(ret)
            self.assertFalse(sw.closed)
            self.assertFalse(buf.closed)
        finally:
            sw.close()

    def test_missing_arg_raises(self):
        """Calling close_streams without v raises TypeError."""
        with self.assertRaises(TypeError):
            StreamWrapper.close_streams()


if __name__ == "__main__":
    run_tests()
