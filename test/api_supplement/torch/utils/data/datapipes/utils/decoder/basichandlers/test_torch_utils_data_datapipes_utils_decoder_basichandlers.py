# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.datapipes.utils.decoder.basichandlers 接口功能正确性
API 名称：torch.utils.data.datapipes.utils.decoder.basichandlers
API 签名：torch.utils.data.datapipes.utils.decoder.basichandlers(extension, data)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 未匹配返回 None 与有效扩展名非空解码       | 已覆盖（unknown_extension、decode_txt_and_json）       |
| 枚举选项         | txt/json/cls/pyd/pickle 扩展名 | 已覆盖（decode_txt_and_json、decode_cls_int、decode_pyd_and_pickle） |
| 参数类型         | 字符串扩展名与 bytes 原始字节数据          | 已覆盖（decode_txt_and_json、decode_cls_int、decode_pyd_and_pickle） |
| 传参与不传参     | 可选默认参数省略与显式传入                 | 未覆盖（extension 与 data 均为必填位置参数）           |
| 等价类/边界值    | cls 非法整数字节返回 None 与未知扩展名     | 已覆盖（decode_cls_int、unknown_extension）            |
| 正常传参场景     | 文本、JSON、整数分类与 pickle 正常解码     | 已覆盖（decode_txt_and_json、decode_cls_int、decode_pyd_and_pickle） |
| 异常传参场景     | 缺少必填参数 extension 或 data             | 已覆盖（missing_args_raises）                          |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（纯字节解码器函数，不接收 Tensor 输入）         |

未覆盖项及原因：
- 传参与不传参：extension 与 data 均为必填位置参数，无可选默认参数。
- 混合设备类型：纯字节流解码辅助函数，不涉及 Tensor 与设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import json
import os
import pickle
import unittest

import torch
from torch.utils.data.datapipes.utils.decoder import basichandlers

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
class TestBasichandlers(TestCase):
    """Test cases for torch.utils.data.datapipes.utils.decoder.basichandlers."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_decode_txt_and_json(self):
        """Decodes UTF-8 bytes for txt and JSON bytes for json."""
        txt_result = basichandlers("txt", "hello world".encode("utf-8"))
        self.assertEqual(txt_result, "hello world")

        payload = {"key": "value", "nums": [1, 2, 3]}
        json_result = basichandlers("json", json.dumps(payload).encode("utf-8"))
        self.assertEqual(json_result, payload)

    def test_decode_cls_int(self):
        """Decodes integer bytes for cls and returns None for non-integer bytes."""
        self.assertEqual(basichandlers("cls", b"42"), 42)
        self.assertIsNone(basichandlers("cls", b"not_an_int"))

    def test_decode_pyd_and_pickle(self):
        """Decodes pickle bytes for pyd and pickle extensions."""
        obj = {"a": [1, 2], "b": (3, 4)}
        pyd_result = basichandlers("pyd", pickle.dumps(obj))
        self.assertEqual(pyd_result, obj)

        lst = [10, 20, 30]
        pkl_result = basichandlers("pickle", pickle.dumps(lst))
        self.assertEqual(pkl_result, lst)
        self.assertEqual(basichandlers("text", b"abc"), "abc")
        self.assertEqual(basichandlers("jsn", b'{"k": 1}'), {"k": 1})

    def test_unknown_extension(self):
        """Unknown extension returns None without raising."""
        self.assertIsNone(basichandlers("unknown_ext", b"raw bytes"))

    def test_missing_args_raises(self):
        """Calling basichandlers with missing arguments raises TypeError."""
        with self.assertRaises(TypeError):
            basichandlers()
        with self.assertRaises(TypeError):
            basichandlers("txt")


if __name__ == "__main__":
    run_tests()
