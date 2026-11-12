# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.swa_utils.AveragedModel.share_memory 接口功能正确性
API 名称：torch.optim.swa_utils.AveragedModel.share_memory
API 签名：share_memory(self) -> Self

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 返回模型自身以支持链式调用 | 已覆盖（test_share_memory_returns_self/test_share_memory_marks_all_cpu_tensors_shared） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 无参方法 | 已覆盖（全部用例无参调用） |
| 传参与不传参 | 无参调用 | 已覆盖 |
| 等价类/边界值 | update 前/后调用均可；n_averaged buffer 亦进入共享内存；数值在迁移后不变 |
已覆盖（share_memory_marks_all_cpu_tensors_shared/share_memory_after_update_still_updatable） |
| 正常传参场景 | share_memory() 返回同一实例；CPU 驻留参数/buffer is_shared()=True 且数值不变； |
NPU 驻留路径实测被 torch_npu 拦截抛 RuntimeError 且模块保持可用 |
已覆盖（share_memory_returns_self/share_memory_marks_all_cpu_tensors_shared/share_memory_device_resident_path） |
| 异常传参场景 | 无参方法，无异常入参路径；NPU 驻留调用的 RuntimeError 属环境限制（按实测锁定） | 已覆盖（test_share_memory_device_resident_path） |
| 混合设备类型 | CPU 驻留验证 is_shared；设备驻留验证调用契约与数值完整性 |
已覆盖（share_memory_marks_all_cpu_tensors_shared/share_memory_device_resident_path） |

未覆盖项及原因：
- 本接口本质为优化器/调度器/状态工具的框架行为，涉及浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。
- is_shared()=True 断言仅在 CPU 驻留张量上成立（API 语义为进程间共享内存；CUDA 文档注明对 CUDA 张量为 no-op）。
- NPU 驻留路径实测为 torch_npu 直接拦截：调用抛 RuntimeError("Module.share_memory ... is not supported in npu.")，
  无法走「可调用+数值不变」路径，故按实测断言该异常并验证失败调用后模块仍可用（test_share_memory_device_resident_path）。
"""
import os
import unittest

import torch
import torch.optim.swa_utils as swa_utils

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 条件导入 torch_npu，并用 torch.npu.is_available() 探测可用 NPU。
DEVICE_NAME = None
try:
    import torch_npu  # noqa: F401
    if torch.npu.is_available():
        DEVICE_NAME = 'npu'
except (ImportError, AttributeError):
    pass
if DEVICE_NAME is None:
    DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestAveragedModelshare_memory(TestCase):
    """Test cases for torch.optim.swa_utils.AveragedModel.share_memory (NPU/CUDA portable).

    AveragedModel inherits nn.Module.share_memory, i.e.
    ``self._apply(lambda t: t.share_memory_())`` applied to every parameter and
    buffer of the whole averaged module. Empirical contract (torch 2.11,
    probed on NPU):
      - returns self and, for CPU-resident tensors, flips is_shared() to
        True on every parameter AND the n_averaged buffer while preserving
        values through the storage migration;
      - update_parameters before/after share_memory keeps working and keeps
        advancing the counter;
      - on NPU-resident modules torch_npu blocks the call outright with
        RuntimeError("Module.share_memory ... is not supported in npu."),
        leaving the module fully usable afterwards (CUDA documents
        share_memory_() as a no-op for CUDA tensors, so the accelerator path
        branches on the probed device behaviour instead of is_shared()).
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_share_memory_returns_self(self):
        """AveragedModel.share_memory returns the module itself."""
        am = torch.optim.swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        result = am.share_memory()
        self.assertIs(result, am)

    def test_share_memory_marks_all_cpu_tensors_shared(self):
        """CPU-resident params and the n_averaged buffer become shared, values intact."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        with torch.no_grad():
            am.module.weight.fill_(1.5)
            am.module.bias.fill_(-0.5)
        weight_before = am.module.weight.detach().clone()
        bias_before = am.module.bias.detach().clone()
        n_before = int(am.n_averaged)

        result = am.share_memory()
        self.assertIs(result, am, "share_memory must return self for chaining")
        # Every parameter of the deep-copied module enters shared memory.
        self.assertTrue(am.module.weight.is_shared(),
                        "CPU parameter storage must be shared after share_memory()")
        self.assertTrue(am.module.bias.is_shared())
        # The registered buffer joins the migration too.
        self.assertTrue(am.n_averaged.is_shared(),
                        "n_averaged buffer must be shared as well")
        # Values survive the storage move to shared memory.
        torch.testing.assert_close(am.module.weight.detach(), weight_before,
                                   msg="weight values must be unchanged")
        torch.testing.assert_close(am.module.bias.detach(), bias_before,
                                   msg="bias values must be unchanged")
        self.assertEqual(int(am.n_averaged), n_before)

    def test_share_memory_after_update_still_updatable(self):
        """update_parameters and share_memory compose in both orders (CPU path)."""
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        am = swa_utils.AveragedModel(model)
        am.update_parameters(model)  # first fold copies the 1.0 weights
        self.assertEqual(int(am.n_averaged), 1)
        am.share_memory()
        # The counter itself lives in shared memory and keeps counting.
        self.assertTrue(am.n_averaged.is_shared(),
                        "the counter must be shared after update+share")
        # Averaging continues to work on top of the shared storages.
        with torch.no_grad():
            model.weight.fill_(2.0)
        am.update_parameters(model)
        self.assertEqual(int(am.n_averaged), 2)
        expected = (1.0 + 2.0) / 2  # running mean over the two folded weights
        torch.testing.assert_close(am.module.weight.detach(), torch.full((2, 2), expected),
                                   msg="running mean must continue after share_memory()")

    def test_share_memory_device_resident_path(self):
        """Device-resident module: NPU raises RuntimeError; CUDA/CPU succeed intact."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2).to(self.device))
        with torch.no_grad():
            am.module.weight.fill_(1.5)
        weight_before = am.module.weight.detach().clone().cpu()

        if self.device_name == 'npu':
            # Probed on NPU: torch_npu blocks nn.Module.share_memory on
            # NPU-resident tensors with RuntimeError (PTA feature not
            # supported), so no callable+is_shared contract exists here.
            with self.assertRaises(RuntimeError):
                am.share_memory()
            # The failed call must leave the module untouched and usable.
            torch.testing.assert_close(am.module.weight.detach().cpu(), weight_before,
                                       msg="a blocked share_memory must not corrupt values")
            model = torch.nn.Linear(2, 2, bias=False).to(self.device)
            with torch.no_grad():
                model.weight.fill_(1.5)
            am.update_parameters(model)
            self.assertEqual(
                int(am.n_averaged), 1,
                "the module must remain updatable after the blocked call")
        elif self.device_name == 'cpu':
            ret = am.share_memory()
            self.assertIs(ret, am)
            self.assertTrue(am.module.weight.is_shared())
            torch.testing.assert_close(am.module.weight.detach().cpu(), weight_before,
                                       msg="values must survive the shared-memory migration")
        else:
            # CUDA documents share_memory_() as a no-op for CUDA tensors, so
            # only callability and value identity are portable assertions.
            ret = am.share_memory()
            self.assertIs(ret, am)
            torch.testing.assert_close(am.module.weight.detach().cpu(), weight_before,
                                       msg="CUDA no-op share must preserve values")


if __name__ == "__main__":
    run_tests()
