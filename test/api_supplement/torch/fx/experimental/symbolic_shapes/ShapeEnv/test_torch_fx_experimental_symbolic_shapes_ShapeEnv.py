# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.ShapeEnv 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.ShapeEnv
API 签名：ShapeEnv(*, should_record_events=None, tracked_fakes=None, **kwargs)

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | tracked_fakes/co_fields 的 None/空/非空值和独立空状态 |
| 枚举选项 | 构造策略 False/True；should_record_events、duck_shape=None/False/True |
| 参数类型 | 策略布尔值、tracked_fakes 列表和 co_fields 字典 |
| 传参与不传参 | 默认构造、显式策略两值，recording/duck_shape=None 和列表/字典显式值 |
| 等价类/边界值 | 环境独立、事件记录开关和列表/字典引用身份 |
| 正常传参场景 | 独立环境中的实际符号及 guard 状态 |
| 异常传参场景 | 未知构造策略关键字 TypeError |
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
from torch.fx.experimental.symbolic_shapes import guard_scalar


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


    def test_shapeenv_environments_keep_symbol_and_guard_states_independent(self):
        first = self._env()
        second = self._env()
        value = self._backed(first, 7)
        self.assertEqual(guard_scalar(value), 7)
        self.assertTrue(first.guards)
        self.assertEqual(second.guards, [])
        self.assertEqual(second.backed_var_to_val, {})

    def test_shapeenv_explicit_policy_and_unbacked_allocation(self):
        env = self._env(duck_shape=False, specialize_zero_one=False)
        self.assertFalse(env.duck_shape)
        self.assertFalse(env.specialize_zero_one)
        value = env.create_unbacked_symint()
        self.assertIsInstance(value, torch.SymInt)
        self.assertIsNone(value.node.hint)
        self.assertIn(value.node.expr, env.pending_fresh_unbacked_symbols)
        self.assertEqual(env.guards, [])


    def test_parameter_all_constructor_policy_options_and_recording_choices(self):
        from dataclasses import fields
        from torch.fx.experimental import _config
        from torch.fx.experimental.symbolic_shapes import ShapeEnvSettings

        with _config.patch(translation_validation=False):
            for recording in (None, False, True):
                for tracked in (None, [], [object()]):
                    env = self._env(should_record_events=recording, tracked_fakes=tracked)
                    self.assertIs(env.tracked_fakes, tracked)
                    self.assertIs(env.should_record_events, recording is True)
                    self.assertEqual(len(env.events), 1 if recording else 0)
            for setting in fields(ShapeEnvSettings):
                for value in (False, True):
                    with self.subTest(setting=setting.name, value=value):
                        env = self._env(**{setting.name: value})
                        self.assertIs(getattr(env.settings, setting.name), value)
            with _config.patch(use_duck_shape=False):
                self.assertFalse(self._env(duck_shape=None).duck_shape)
            with _config.patch(use_duck_shape=True):
                self.assertTrue(self._env(duck_shape=None).duck_shape)
        for source_fields in (None, {}, {"co_name": "test_function"}):
            env = self._env(co_fields=source_fields)
            self.assertEqual(env.co_fields, source_fields or {})
            if source_fields:
                self.assertIs(env.co_fields, source_fields)
        with self.assertRaises(TypeError):
            ShapeEnv(unknown_policy=True)


if __name__ == "__main__":
    run_tests()
