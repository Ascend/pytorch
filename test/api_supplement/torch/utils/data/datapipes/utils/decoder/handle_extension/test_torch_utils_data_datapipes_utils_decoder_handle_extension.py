# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.datapipes.utils.decoder.handle_extension 接口功能正确性
API 名称：torch.utils.data.datapipes.utils.decoder.handle_extension
API 签名：torch.utils.data.datapipes.utils.decoder.handle_extension(extensions, f)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 空扩展名规格与非空扩展名匹配               | 已覆盖（empty_extensions、single_extension_match）     |
| 枚举选项         | 接口枚举类参数覆盖                         | 未覆盖（接口无枚举类参数）                             |
| 参数类型         | 字符串扩展名规格与可调用处理函数 f         | 已覆盖（single_extension_match、dotted_extension）     |
| 传参与不传参     | 可选默认参数省略与显式传入                 | 未覆盖（extensions 与 f 均为必填位置参数）             |
| 等价类/边界值    | 大小写不敏感、多段点号后缀与空规格不匹配   | 已覆盖（case_insensitive、dotted_extension、empty_extensions） |
| 正常传参场景     | 命中扩展名调用 f(data) 与未命中返回 None   | 已覆盖（single_extension_match、case_insensitive、dotted_extension） |
| 异常传参场景     | 缺少必填参数 extensions 或 f               | 已覆盖（missing_args_raises）                          |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（纯扩展名分发高阶函数，不接收 Tensor 输入）     |

未覆盖项及原因：
- 枚举选项：接口无枚举类参数。
- 传参与不传参：extensions 与 f 均为必填位置参数，无可选默认参数。
- 混合设备类型：纯字符串扩展名匹配路由函数，不涉及 Tensor 与设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.data.datapipes.utils.decoder import handle_extension

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
class TestHandleExtension(TestCase):
    """Test cases for torch.utils.data.datapipes.utils.decoder.handle_extension."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_single_extension_match(self):
        """Matching key invokes wrapped handler f(data); non-matching returns None."""
        calls = []

        def decode_utf8(data):
            calls.append(data)
            return data.decode("utf-8")

        handler = handle_extension("txt", decode_utf8)
        self.assertTrue(callable(handler))
        self.assertEqual(handler("sample.txt", b"hello"), "hello")
        self.assertEqual(calls, [b"hello"])
        self.assertIsNone(handler("sample.json", b"{}"))
        self.assertEqual(len(calls), 1)

    def test_case_insensitive(self):
        """Extension matching is case insensitive on both spec and key."""
        handler = handle_extension("TXT", lambda data: data.upper())
        self.assertEqual(handler("FILE.Txt", b"abc"), b"ABC")
        self.assertIsNone(handler("FILE.BIN", b"abc"))

    def test_dotted_extension(self):
        """Multi-component dotted extension matches only full suffix components."""
        handler = handle_extension("seg.jpg", lambda data: ("seg", data))
        self.assertEqual(handler("img.seg.jpg", b"123"), ("seg", b"123"))
        self.assertIsNone(handler("img.jpg", b"123"))
        self.assertIsNone(handler("jpg", b"123"))

    def test_empty_extensions(self):
        """Empty extension specification never matches and returns None."""
        handler = handle_extension("", lambda data: data)
        self.assertIsNone(handler("file.txt", b"abc"))

    def test_missing_args_raises(self):
        """Calling handle_extension with missing arguments raises TypeError."""
        with self.assertRaises(TypeError):
            handle_extension()
        with self.assertRaises(TypeError):
            handle_extension("txt")


if __name__ == "__main__":
    run_tests()
