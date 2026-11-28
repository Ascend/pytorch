# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_verbose 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_verbose
API 签名：ShapeEnv.produce_guards_verbose(self, placeholders, sources, source_ref=lambda n: n.name, *, guards=None,
    input_contexts=None, equalities_inputs=None, _simplified=False, ignore_static=True, langs=('python',
    'verbose_python')) -> list[_ShapeGuardsHelper]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 空输入、guards=None/空/非空、contexts/equalities=None/对象 |
| 枚举选项 | python/verbose_python/cpp；_simplified 和 ignore_static 两值 |
| 参数类型 | 符号/FakeTensor、Source、source_ref、guard 列表、符号上下文和 EqualityConstraint |
| 传参与不传参 | source_ref 默认/自定义，guards/contexts/equalities 默认/显式值和语言/布尔选项 |
| 等价类/边界值 | 空输入、guard 过滤与值域检查、静态尺寸和有效/无效运行时值 |
| 正常传参场景 | 真实 guard 序列化后对有效/无效运行时值求值 |
| 异常传参场景 | placeholder/source 长度不匹配 AssertionError；未知语言 NotImplementedError |
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

from torch._dynamo.source import LocalSource
from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _env(self, **kwargs):
        env = ShapeEnv(**kwargs)
        self.addCleanup(env.cleanup)
        return env

    def _backed(self, env, value=7, name="value"):
        return env.create_unspecified_symint_and_symbol(value, LocalSource(name), DimDynamic.DYNAMIC)


    def test_produce_guards_verbose_generated_guards_accept_and_reject_runtime_values(self):
        env = self._env()
        value = self._backed(env, 7, "t0")
        self.assertTrue(bool(value > 3))
        outputs = env.produce_guards_verbose([value], [LocalSource("t0")])
        self.assertEqual(len(outputs), 2)
        self.assertTrue(outputs[0].exprs)
        self.assertTrue(outputs[1].exprs)
        for output in outputs:
            self.assertTrue(all(env.evaluate_guards_expression(item, [7]) for item in output.exprs))
            self.assertFalse(all(env.evaluate_guards_expression(item, [2]) for item in output.exprs))

    def test_produce_guards_verbose_empty_inputs_and_explicit_options(self):
        env = self._env()
        value = self._backed(env, 7, "t0")
        outputs = env.produce_guards_verbose([], [], langs=("python",))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].exprs, [])
        with self.assertRaises(AssertionError):
            env.produce_guards_verbose([value], [])


    def test_parameter_all_guard_languages_custom_source_and_unknown_language(self):
        import re

        env = self._env()
        value = self._backed(env, 7, "t0")
        self.assertTrue(bool(value > 3))
        seen = []

        def source_ref(source):
            seen.append(source.name)
            return "guard_input"

        for language in ("python", "verbose_python", "cpp"):
            with self.subTest(language=language):
                outputs = env.produce_guards_verbose([value], [LocalSource("t0")],
                                                     source_ref=source_ref, langs=(language,))
                self.assertEqual(len(outputs), 1)
                self.assertTrue(outputs[0].exprs)
                verdicts = {2: [], 7: []}
                expressions = outputs[0].exprs
                for expression in expressions:
                    executable = expression.split("#", 1)[0]
                    if language == "cpp":
                        executable = re.sub(r"(?<=\d)L\b", "", executable)
                        executable = executable.replace("&&", " and ").replace("||", " or ")
                    identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", executable)) - {"and", "or"}
                    self.assertEqual(len(identifiers), 1)
                    identifier = next(iter(identifiers))
                    for current in (2, 7):
                        verdicts[current].append(eval(executable, {"__builtins__": {}}, {identifier: current}))
                self.assertTrue(all(verdicts[7]))
                self.assertFalse(all(verdicts[2]))
        self.assertIn("L['t0']", seen)
        with self.assertRaisesRegex(NotImplementedError, "Unknown lang"):
            env.produce_guards_verbose([value], [LocalSource("t0")], langs=("invalid",))

    def test_parameter_explicit_guard_filter_contexts_equalities_and_simplification(self):
        from torch.fx.experimental.symbolic_shapes import EqualityConstraint, SymIntSymbolicContext

        env = self._env()
        value = self._backed(env, 7, "t0")
        self.assertTrue(bool(value > 3))
        equalities = EqualityConstraint(warn_only=False, source_pairs=[], derived_equalities=[],
                                        phantom_symbols=[], relaxed_sources=set())
        for simplified in (False, True):
            for contexts in (None, [None], [SymIntSymbolicContext(None)]):
                for equality in (None, equalities):
                    with self.subTest(simplified=simplified, contexts=contexts, equality=equality):
                        output = env.produce_guards_verbose(
                            [value], [LocalSource("t0")], guards=list(env.guards), input_contexts=contexts,
                            equalities_inputs=equality, _simplified=simplified, ignore_static=False,
                            langs=("python",))[0]
                        self.assertTrue(all(env.evaluate_guards_expression(expr, [7]) for expr in output.exprs))
                        self.assertFalse(all(env.evaluate_guards_expression(expr, [2]) for expr in output.exprs))
        filter_env = self._env()
        parity_value = self._backed(filter_env, 7, "t0")
        self.assertTrue(bool(parity_value % 2 == 1))
        guard_count = len(filter_env.guards)
        unfiltered = filter_env.produce_guards_verbose([parity_value], [LocalSource("t0")],
                                                       guards=None, langs=("python",))[0]
        filtered = filter_env.produce_guards_verbose([parity_value], [LocalSource("t0")],
                                                     guards=[], langs=("python",))[0]
        self.assertTrue(all(filter_env.evaluate_guards_expression(expr, [7]) for expr in unfiltered.exprs))
        self.assertFalse(all(filter_env.evaluate_guards_expression(expr, [2]) for expr in unfiltered.exprs))
        for current in (2, 7):
            self.assertTrue(all(filter_env.evaluate_guards_expression(expr, [current]) for expr in filtered.exprs))
        self.assertEqual(len(filter_env.guards), guard_count)

    def test_parameter_ignore_static_false_checks_static_tensor_dimensions(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        env = self._env()
        mode = FakeTensorMode(shape_env=env)
        real = torch.ones(2, 3, device=self.device)
        fake = mode.from_tensor(real, static_shapes=True)
        strict = env.produce_guards_verbose([fake], [LocalSource("t0")],
                                            ignore_static=False, langs=("python",))[0].exprs
        relaxed = env.produce_guards_verbose([fake], [LocalSource("t0")],
                                             ignore_static=True, langs=("python",))[0].exprs
        self.assertGreater(len(strict), len(relaxed))
        self.assertTrue(all(env.evaluate_guards_expression(expr, [real]) for expr in strict))
        changed = torch.ones(4, 3, device=self.device)
        self.assertFalse(all(env.evaluate_guards_expression(expr, [changed]) for expr in strict))


if __name__ == "__main__":
    run_tests()
