# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.unification.multipledispatch.dispatcher.restart_ordering 接口功能正确性
API 名称：torch.fx.experimental.unification.multipledispatch.dispatcher.restart_ordering
API 签名：restart_ordering(on_ambiguity=ambiguity_warn) -> None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | on_ambiguity 省略、None 和回调 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 回调及 None；弃用空操作不读取回调参数 |
| 传参与不传参 | on_ambiguity 默认值和显式 None/回调 |
| 等价类/边界值 | 重复调用与不同类型路由 |
| 正常传参场景 | FutureWarning、回调不执行及原有分派结果保持 |
| 异常传参场景 | 未覆盖：没有声明额外参数校验 |
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

from torch.fx.experimental.unification.multipledispatch.dispatcher import restart_ordering
import warnings
from torch.fx.experimental.unification.multipledispatch.dispatcher import Dispatcher


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_restart_ordering_deprecation_preserves_dispatch_semantics(self):
        dispatcher = Dispatcher("graph_rule")
        dispatcher.add((int,), lambda value: value + 4)
        snapshot = dict(dispatcher.funcs)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            restart_ordering()
        self.assertEqual(len(captured), 1)
        self.assertTrue(issubclass(captured[0].category, FutureWarning))
        self.assertIn("deprecated", str(captured[0].message))
        self.assertEqual(dispatcher.funcs, snapshot)
        self.assertEqual(dispatcher(3), 7)

    def test_restart_ordering_repeated_calls_keep_type_routing(self):
        dispatcher = Dispatcher("repeated_rule")
        dispatcher.add((object,), lambda value: "object")
        dispatcher.add((int,), lambda value: "integer")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            restart_ordering()
            restart_ordering()
        self.assertEqual(dispatcher(3), "integer")
        self.assertEqual(dispatcher("x"), "object")


    def test_parameter_explicit_callback_and_none_preserve_deprecated_noop_contract(self):
        calls = []

        def callback(*args):
            calls.append(args)

        for supplied in (None, callback):
            with self.subTest(supplied=supplied):
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    restart_ordering(on_ambiguity=supplied)
                self.assertEqual(len(captured), 1)
                self.assertTrue(issubclass(captured[0].category, FutureWarning))
                self.assertIn("deprecated", str(captured[0].message))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    run_tests()
