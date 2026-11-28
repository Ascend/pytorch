# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.operator_schemas.normalize_function 接口功能正确性
API 名称：torch.fx.operator_schemas.normalize_function
API 签名：normalize_function(target, args, kwargs=None, arg_types=None, kwarg_types=None,
    normalize_to_only_use_kwargs=False) -> torch.fx.operator_schemas.ArgsKwargsPair | None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | kwargs、arg_types、kwarg_types 的 None/空/非空值 |
| 枚举选项 | normalize_to_only_use_kwargs=False/True |
| 参数类型 | Python/Torch 函数、Node 参数、类型元组/字典及 Tensor/int/float/str 重载 |
| 传参与不传参 | kwargs/类型提示省略、None/显式值和规范化开关两值 |
| 等价类/边界值 | 默认值填充、仅 kwargs 调用、重载消歧和 Node 引用保持 |
| 正常传参场景 | 默认参数填充、kwargs 规范化及执行语义 |
| 异常传参场景 | 重载歧义 RuntimeError；非 callable 目标 AssertionError |
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

from torch.fx.operator_schemas import normalize_function


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_normalize_function_python_defaults_and_kwargs_only_normalization_preserve_result(self):
        def function(x, y=2, *, scale=3):
            return (x + y) * scale

        result = normalize_function(function, (4,), normalize_to_only_use_kwargs=True)
        self.assertEqual(result.args, ())
        self.assertEqual(result.kwargs, {"x": 4, "y": 2, "scale": 3})
        self.assertEqual(function(*result.args, **result.kwargs), 18)
        explicit = normalize_function(function, (4, 5), {"scale": 2})
        self.assertEqual(function(*explicit.args, **explicit.kwargs), 18)

    def test_normalize_function_explicit_operator_overload_preserves_node_identity_and_defaults(self):
        graph = torch.fx.Graph()
        a, b = graph.placeholder("a"), graph.placeholder("b")
        result = normalize_function(torch.ops.aten.add.Tensor, (a, b), {"alpha": 2})
        self.assertEqual(result.args, (a, b))
        self.assertEqual(result.kwargs, {"alpha": 2})
        default = normalize_function(torch.ops.aten.add.Tensor, (a, b))
        self.assertEqual(default.kwargs, {"alpha": 1})


    def test_parameter_argument_types_disambiguate_overloaded_torch_function(self):
        graph = torch.fx.Graph()
        first, second = graph.placeholder("first"), graph.placeholder("second")
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            normalize_function(torch.add, (first, second))
        for other_type in (torch.Tensor, int, float):
            with self.subTest(other_type=other_type):
                result = normalize_function(torch.add, (first, second),
                                            arg_types=(torch.Tensor, other_type), kwarg_types=None)
                self.assertEqual(result.args, (first, second))
                self.assertEqual(result.kwargs, {"alpha": 1})
        result = normalize_function(torch.add, (), {"input": first, "other": second},
                                    arg_types=(), kwarg_types={"input": torch.Tensor, "other": torch.Tensor})
        self.assertEqual(result.args, ())
        self.assertEqual(result.kwargs, {"input": first, "other": second, "alpha": 1})
        strings = normalize_function(torch.add, (first, second), arg_types=(str, str))
        self.assertEqual(strings.args, (first, second))
        self.assertEqual(strings.kwargs, {})

    def test_parameter_none_and_empty_kwargs_and_both_normalization_options(self):
        def function(value, amount=2):
            return value + amount

        for kwargs in (None, {}):
            for only_kwargs in (False, True):
                with self.subTest(kwargs=kwargs, only_kwargs=only_kwargs):
                    result = normalize_function(function, (4,), kwargs=kwargs, arg_types=None,
                                                kwarg_types=None, normalize_to_only_use_kwargs=only_kwargs)
                    self.assertEqual(function(*result.args, **result.kwargs), 6)
                    if only_kwargs:
                        self.assertEqual(result.args, ())
                        self.assertEqual(result.kwargs, {"value": 4, "amount": 2})
                    else:
                        self.assertEqual(result.args, (4,))
                        self.assertEqual(result.kwargs, {"amount": 2})
        with self.assertRaises(AssertionError):
            normalize_function(None, ())


if __name__ == "__main__":
    run_tests()
