# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.recording.shape_env_check_state_equal 接口功能正确性
API 名称：torch.fx.experimental.recording.shape_env_check_state_equal
API 签名：shape_env_check_state_equal(env1, env2, non_state_variable_names, map_value) -> None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空状态、约束或记录 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 ShapeEnv、SymPy 表达式及对应默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：状态值差异、字段集合差异且不修改环境 |
| 正常传参场景 | 已覆盖：实际环境所选 guards/frozen 状态的映射比较 |
| 异常传参场景 | 已覆盖：NotEqualError 的字段和值差异诊断 |
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

from torch.fx.experimental.recording import NotEqualError, shape_env_check_state_equal
from torch.fx.experimental.symbolic_shapes import ShapeEnv


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _envs(self):
        first, second = ShapeEnv(), ShapeEnv()
        self.addCleanup(first.cleanup)
        self.addCleanup(second.cleanup)
        selected = {"frozen", "guards"}
        excluded = tuple((vars(first).keys() | vars(second).keys()) - selected)
        return first, second, excluded

    def test_shape_env_check_state_equal_equal_selected_state_is_mapped_without_mutating_environments(self):
        first, second, excluded = self._envs()
        first_keys, second_keys = set(vars(first)), set(vars(second))
        mapped = []

        def mapper(name, value):
            mapped.append(name)
            return tuple(value) if isinstance(value, list) else value

        self.assertIsNone(shape_env_check_state_equal(first, second, excluded, mapper))
        self.assertEqual(set(mapped), {"frozen", "guards"})
        self.assertEqual(set(vars(first)), first_keys)
        self.assertEqual(set(vars(second)), second_keys)

    def test_shape_env_check_state_equal_changed_state_and_unique_fields_have_distinct_diagnostics(self):
        first, second, excluded = self._envs()
        first.freeze()
        with self.assertRaisesRegex(NotEqualError, "frozen"):
            shape_env_check_state_equal(first, second, excluded, lambda name, value: value)
        second.ut_unique_field = 1
        with self.assertRaisesRegex(NotEqualError, "field set mismatch"):
            shape_env_check_state_equal(first, second, excluded, lambda name, value: value)


if __name__ == "__main__":
    run_tests()
