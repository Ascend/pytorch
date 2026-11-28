# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.proxy_tensor.fake_signature 接口功能正确性
API 名称：torch.fx.experimental.proxy_tensor.fake_signature
API 签名：fake_signature(fn, nargs) -> Callable[_P, R]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的无 mode、空状态或空参数 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 tracer/mode、Tensor 或符号及默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：零参数与参数个数错误 |
| 正常传参场景 | 已覆盖：varargs 转固定参数签名和实际转发 |
| 异常传参场景 | 已覆盖：缺失位置参数 TypeError |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

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

from torch.fx.experimental import proxy_tensor as pt
import inspect


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_fake_signature_varargs_callable_receives_fixed_signature_arguments(self):
        calls = []

        def function(*values):
            calls.append(values)
            return sum(values) + 1

        wrapped = pt.fake_signature(function, 2)
        self.assertEqual(len(inspect.signature(wrapped).parameters), 2)
        self.assertEqual(wrapped(3, 4), 8)
        self.assertEqual(calls, [(3, 4)])
        with self.assertRaises(TypeError):
            wrapped(3)

    def test_fake_signature_zero_argument_signature_is_traceable(self):
        wrapped = pt.fake_signature(lambda: 7, 0)
        self.assertEqual(len(inspect.signature(wrapped).parameters), 0)
        self.assertEqual(wrapped(), 7)
        module = torch.fx.symbolic_trace(wrapped)
        self.assertEqual(module(), 7)


if __name__ == "__main__":
    run_tests()
