# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.PReLU.reset_parameters 接口功能正确性
API 名称：torch.nn.PReLU.reset_parameters
API 签名：reset_parameters(self) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；对默认与显式 init 值的重置结果分别精确断言 | 已覆盖（test_reset_parameters_init_default 等） |
| 枚举选项 | init=0.25（默认）/0.5 多个取值逐个断言 | 已覆盖（test_reset_parameters_init_explicit） |
| 参数类型 | weight 为 float32 Parameter，重置后 shape 与 dtype 不变 | 已覆盖（test_reset_parameters_preserves_shape_and_grad） |
| 传参与不传参 | reset_parameters 无参；按 __init__ 使用的 init 值重置 | 已覆盖（test_reset_parameters_init_default 等） |
| 等价类/边界值 | num_parameters=1/2/3/4 多个尺寸下重置值精确对照 | 已覆盖（test_reset_parameters_init_explicit 等） |
| 正常传参场景 | __init__ 经 reset_parameters 完成初始填充；重置恢复 init 值 | 已覆盖（test_reset_parameters_restores_init 等） |
| 异常传参场景 | reset_parameters 无参无校验路径 | 未覆盖（无稳定异常） |
| 混合设备类型 | 设备侧构造的 PReLU 重置后与 CPU 参考值一致且张量在设备上 | 已覆盖（test_reset_parameters_device_consistency） |

未覆盖项及原因：
- reset_parameters 以构造时传入的 init 值做 torch.nn.init.constant_ 填充，
  实测 torch 2.15.0（wheel 源码 torch.nn.init.constant_(self.weight, self.init)）：
  init=0.5 实例重置后恢复 0.5，缺省 init=0.25 重置恢复 0.25；非随机分布、非固定 0.25。
  重置按固定路径填充、无校验分支，故无异常路径。
- 本文件同时看护别名路径 torch.nn.modules.activation.PReLU.reset_parameters：
  torch.nn.PReLU 与 torch.nn.modules.activation.PReLU 为同一类对象，
  test_dual_import_paths_same_object 断言其同一性。
- 设备构造测试在 NPU/GPU 上真实分配与填充 weight；无可用设备时按规范抛 RuntimeError。

注意：浮点 Tensor 对照使用 torch.testing.assert_close（设备无关）；
     常量填充值为精确可表示数值（0.25/0.5），同时用 assert_close 断言张量相等。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"

DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestPReLUResetParameters(TestCase):
    """Comprehensive cases for torch.nn.PReLU.reset_parameters (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.PReLU and torch.nn.modules.activation.PReLU are the same class."""
        self.assertIs(torch.nn.PReLU, torch.nn.modules.activation.PReLU)

    def test_reset_parameters_init_default(self):
        """Constructor fills weight with the default init value 0.25 via reset_parameters."""
        m = torch.nn.PReLU()
        self.assertEqual(tuple(m.weight.shape), (1,))
        torch.testing.assert_close(m.weight.detach().cpu(), torch.full((1,), 0.25))

    def test_reset_parameters_init_explicit(self):
        """Explicit init value is applied to every element of weight."""
        m = torch.nn.PReLU(3, init=0.5)
        torch.testing.assert_close(m.weight.detach().cpu(), torch.full((3,), 0.5))

    def test_reset_parameters_restores_init(self):
        """After mutating weight, reset_parameters restores the init value."""
        # Measured on torch 2.15.0: PReLU(3, init=0.5) with weight filled to
        # -1.0 then reset_parameters() -> [0.5, 0.5, 0.5]; reset_parameters
        # runs torch.nn.init.constant_(self.weight, self.init), so the fill
        # value is the constructor's init, not a fixed 0.25.
        m = torch.nn.PReLU(3, init=0.5)
        with torch.no_grad():
            m.weight.fill_(-1.0)
        torch.testing.assert_close(m.weight.detach().cpu(), torch.full((3,), -1.0))
        m.reset_parameters()
        torch.testing.assert_close(m.weight.detach().cpu(), torch.full((3,), 0.5))

    def test_reset_parameters_idempotent(self):
        """Repeated reset_parameters calls keep the same init value."""
        m = torch.nn.PReLU(2, init=0.25)
        m.reset_parameters()
        m.reset_parameters()
        torch.testing.assert_close(m.weight.detach().cpu(), torch.full((2,), 0.25))

    def test_reset_parameters_preserves_shape_and_grad(self):
        """Reset keeps the weight shape and requires_grad state."""
        # Measured on torch 2.15.0: after mul_(3.0) and reset_parameters(),
        # shape stays (4,), requires_grad stays True, values return to 0.25.
        m = torch.nn.PReLU(4, init=0.25)
        self.assertTrue(m.weight.requires_grad)
        with torch.no_grad():
            m.weight.mul_(3.0)
        m.reset_parameters()
        self.assertEqual(tuple(m.weight.shape), (4,))
        self.assertTrue(m.weight.requires_grad)
        torch.testing.assert_close(m.weight.detach().cpu(), torch.full((4,), 0.25))

    def test_reset_parameters_device_consistency(self):
        """Device-constructed PReLU resets in place with values matching the CPU reference."""
        # Measured on torch 2.15.0 (Ascend910B4): PReLU(2, init=0.5,
        # device='npu') after fill_(-2.0) and reset_parameters() -> [0.5, 0.5]
        # with weight still on the device, matching the CPU reference.
        cpu_m = torch.nn.PReLU(2, init=0.5)
        device_m = torch.nn.PReLU(2, init=0.5, device=self.device_name)
        self.assertEqual(device_m.weight.device.type, self.device_name)
        with torch.no_grad():
            device_m.weight.fill_(-2.0)
        device_m.reset_parameters()
        torch.testing.assert_close(device_m.weight.detach().cpu(), torch.full((2,), 0.5))
        torch.testing.assert_close(device_m.weight.detach().cpu(), cpu_m.weight.detach().cpu())


if __name__ == "__main__":
    run_tests()
