# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.compiler.precompile.PrecompileSummary.complete 接口功能正确性
API 名称：torch.compiler.precompile.PrecompileSummary.complete
API 签名：PrecompileSummary.complete: bool（只读属性）

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：空 guard/错误/覆盖列表 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 PrecompileSummary 的 keyword-only 字段与默认值 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：零编译和 guard 泛化风险不影响覆盖完成语义 |
| 正常传参场景 | 已覆盖：覆盖完成判定与所有缺口字段 |
| 异常传参场景 | 未覆盖：没有稳定的异常契约 |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

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

    def test_complete_coverage_gaps_and_zero_compilation_make_summary_incomplete(self):
        self.assertTrue(self._summary().complete)
        for fields in ({"bypassed": ("forward",)}, {"truncated": ("forward (model.py:7)",)},
                       {"uncovered_frames": ("forward",)}, {"capture_errors": ("RuntimeError: error",)},
                       {"guarded_codes": 0}, {"backend_graphs": 0}):
            with self.subTest(fields=fields):
                self.assertFalse(self._summary(**fields).complete)

    def test_complete_guard_generalization_findings_do_not_change_coverage_completeness(self):
        value = self._summary(dropped_guards=(("ID_MATCH", "model"),),
                              risky_dropped_guards=(("ID_MATCH", "model"),), wont_generalize=("count",),
                              policy_dropped_guards=(("TYPE_MATCH", "other"),))
        self.assertTrue(value.complete)


if __name__ == "__main__":
    run_tests()
