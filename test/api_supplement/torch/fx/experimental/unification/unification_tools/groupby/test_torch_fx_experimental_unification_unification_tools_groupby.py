# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.unification.unification_tools.groupby 接口功能正确性
API 名称：torch.fx.experimental.unification.unification_tools.groupby
API 签名：groupby(key, seq) -> dict[object, list[object]]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 空与非空输入 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 回调、字符串/整数/None 成员键、成员键列表和迭代器 |
| 传参与不传参 | 不适用：key 和 seq 均必传 |
| 等价类/边界值 | 多成员键、空键列表、空序列和组内顺序 |
| 正常传参场景 | 回调键、成员键、组内顺序 |
| 异常传参场景 | None 序列 TypeError；缺失成员 KeyError |
| 混合设备类型 | 不适用：本文件为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
DEVICE_NAME = None
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch_npu":
            raise
    else:
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    if DEVICE_NAME is None and torch.cuda.is_available():
        DEVICE_NAME = "cuda"

if DEVICE_NAME == "npu":
    from torch_npu.testing.testcase import TestCase, run_tests
else:
    from unittest import TestCase

    def run_tests():
        unittest.main()

from torch.fx.experimental.unification.unification_tools import groupby


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_groupby_callable_key_preserves_group_order(self):
        self.assertEqual(groupby(len, ["ab", "c", "de", "f"]), {2: ["ab", "de"], 1: ["c", "f"]})
        self.assertEqual(groupby(len, []), {})

    def test_groupby_member_key_and_generator(self):
        rows = [{"kind": "a", "v": 1}, {"kind": "b", "v": 2}, {"kind": "a", "v": 3}]
        self.assertEqual(groupby("kind", rows), {"a": [rows[0], rows[2]], "b": [rows[1]]})
        self.assertEqual(groupby(lambda v: v % 2, iter([1, 2, 3])), {1: [1, 3], 0: [2]})


    def test_parameter_multiple_member_keys_and_none_key(self):
        rows = [{"a": 1, "b": 2}, {"a": 1, "b": 3}, {"a": 1, "b": 2}]
        self.assertEqual(groupby(["a", "b"], iter(rows)), {(1, 2): [rows[0], rows[2]], (1, 3): [rows[1]]})
        self.assertEqual(groupby([], rows), {(): rows})
        self.assertEqual(groupby(None, [{None: "a"}, {None: "a"}]), {"a": [{None: "a"}, {None: "a"}]})
        self.assertEqual(groupby(0, [(1, "a"), (2, "b")]), {1: [(1, "a")], 2: [(2, "b")]})
        with self.assertRaises(TypeError):
            groupby(len, None)
        with self.assertRaises(KeyError):
            groupby("missing", [{}])


if __name__ == "__main__":
    run_tests()
