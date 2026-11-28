# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.compiler.precompile.PrecompileSummary 接口功能正确性
API 名称：torch.compiler.precompile.PrecompileSummary
API 签名：PrecompileSummary(*, frames, resume_functions, guarded_codes, backend_graphs, bypassed=(), truncated=(),
    uncovered_frames=(), eager_fallbacks=(), wont_generalize=(), dropped_guards=(), kept_guards=(),
    risky_dropped_guards=(), policy_dropped_guards=(), dropped_guard_code=(), capture_errors=())

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 空 guard/错误/覆盖列表 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | keyword-only 整数字段、字符串元组、guard 对/三元组和错误元组 |
| 传参与不传参 | 全部可选字段省略、显式空元组和非空元组 |
| 等价类/边界值 | 四个计数字段 0/1/2、列表截断、多行错误、pickle 与哈希 |
| 正常传参场景 | 摘要计数、guard 风险与冻结字段 |
| 异常传参场景 | 未覆盖：没有稳定的异常契约 |
| 混合设备类型 | 不适用：本文件为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
- 图执行方式：本文件验证覆盖报告的数据语义，不代表该 summary 来自本次真实编译产物。
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

from torch.compiler.precompile import PrecompileSummary

import dataclasses


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _summary(self, **changes):
        fields = dict(frames=2, resume_functions=1, guarded_codes=2, backend_graphs=3)
        fields.update(changes)
        return PrecompileSummary(**fields)

    def test_precompilesummary_digest_counts_guard_drops_and_frozen_fields(self):
        value = self._summary(dropped_guards=(("ID_MATCH", "model"),),
                              kept_guards=(("TENSOR_MATCH", "x"),),
                              risky_dropped_guards=(("ID_MATCH", "model"),))
        text = str(value)
        self.assertIn("2 frames (1 from graph breaks)", text)
        self.assertIn("2 guarded codes", text)
        self.assertIn("3 backend graphs", text)
        self.assertIn("dropped guards {'ID_MATCH': 1} (1 kept)", text)
        self.assertIn("RISKY drops", text)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            value.frames = 99

    def test_precompilesummary_digest_truncates_long_lists_and_uses_first_nonempty_error_line(self):
        frames = tuple("frame_" + str(index) for index in range(7))
        value = self._summary(bypassed=frames, capture_errors=("\nRuntimeError: first\nextra", "ValueError: second"))
        text = str(value)
        self.assertIn("7 BYPASSED", text)
        self.assertIn("+2 more", text)
        self.assertNotIn("frame_6", text)
        self.assertIn("2 CAPTURE ERRORS: 'RuntimeError: first'", text)
        self.assertNotIn("ValueError: second", text)


    def test_parameter_all_optional_fields_omitted_explicit_empty_and_nonempty(self):
        import pickle

        fields = {field.name: field for field in dataclasses.fields(PrecompileSummary)}
        defaults = {name: field.default for name, field in fields.items() if field.default is not dataclasses.MISSING}
        default = self._summary()
        explicit = self._summary(**defaults)
        self.assertEqual(default, explicit)
        for name in defaults:
            if name == "dropped_guard_code":
                payload = (("TYPE_MATCH", "source", "source is not None"),)
            elif name.endswith("guards"):
                payload = (("TYPE_MATCH", "source"),)
            else:
                payload = ("frame",)
            with self.subTest(field=name):
                record = self._summary(**{name: payload})
                self.assertIs(getattr(record, name), payload)
                restored = pickle.loads(pickle.dumps(record))
                self.assertEqual(record, restored)
                self.assertEqual(hash(record), hash(restored))
                self.assertEqual(getattr(default, name), ())
        for count in (0, 1, 2):
            record = PrecompileSummary(frames=count, resume_functions=count, guarded_codes=count, backend_graphs=count)
            self.assertEqual((record.frames, record.resume_functions, record.guarded_codes, record.backend_graphs),
                             (count, count, count, count))


if __name__ == "__main__":
    run_tests()
