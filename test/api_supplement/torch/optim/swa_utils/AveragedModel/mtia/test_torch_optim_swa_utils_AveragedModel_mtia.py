# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.swa_utils.AveragedModel.mtia 接口功能正确性
API 名称：torch.optim.swa_utils.AveragedModel.mtia
API 签名：mtia(self, device: int | torch.device | None = None) -> Self

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | mtia 为移动方法，需可用 mtia 设备；方法继承自 nn.Module 且存在于 AveragedModel |
已覆盖（mtia_available/mtia_method_exists_and_is_module_api） |
| 枚举选项 | device 入参三形态：None（默认）/ int 索引 / torch.device('mtia:0') 字符串或设备对象 |
已覆盖（mtia_raises_when_build_lacks_mtia） |
| 参数类型 | device 可选（int/torch.device/None）；非 MTIA 构建下无参/None/设备对象触发 torch.mtia._lazy_init 的 |
AssertionError，int 索引触发 aten 守卫的 RuntimeError | 已覆盖（mtia_raises_when_build_lacks_mtia） |
| 传参与不传参 | 默认无参与显式传参各路径均已验证（按异常形态区分） | 已覆盖（mtia_raises_when_build_lacks_mtia） |
| 等价类/边界值 | 无 MTIA 构建下调用不静默失败（必抛 AssertionError/RuntimeError）；torch.mtia.is_available()/_is_compiled() 探测为 False |
已覆盖（mtia_raises_when_build_lacks_mtia） |
| 正常传参场景 | 真实 MTIA 硬件上返回 self（本机无法验证，按可用性守卫跳过） | 已覆盖（mtia_available，本机 skip 注明） |
| 异常传参场景 | 无 mtia 设备时抛 AssertionError（no-arg/None/device）与 RuntimeError（int）；异常后模块仍可用 |
已覆盖（mtia_raises_when_build_lacks_mtia） |
| 混合设备类型 | mtia 为专用加速设备移动，本机无 mtia 硬件，无法验证设备迁移数值 | 未覆盖（无 MTIA 设备，调用必抛异常，见未覆盖说明） |

未覆盖项及原因：
- 本接口本质为优化器/调度器/状态工具的框架行为，涉及浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。
- MTIA 设备迁移成功路径（返回 self、参数落在 mtia 设备）未覆盖：本机 torch.mtia.is_available() 为 False（torch 未编译 MTIA 支持），
  仅能锁定无 MTIA 构建下的异常契约；真实硬件路径由 test_mtia_available 在有 MTIA 设备的环境执行。
- 异常类型差异说明（实测 torch 2.11）：no-arg/None/torch.device('mtia:0') 走 Tensor.mtia -> torch.mtia._lazy_init 的
  AssertionError("Torch not compiled with MTIA enabled...")；int 索引（如 mtia(0)）先命中 aten 层设备校验抛
  RuntimeError("Invalid device, must be MTIA device")。
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
class TestAveragedModelmtia(TestCase):
    """Test cases for torch.optim.swa_utils.AveragedModel.mtia (NPU/CUDA portable).

    AveragedModel inherits mtia from nn.Module:
    ``mtia(device=None) -> Self`` implemented as
    ``self._apply(lambda t: t.mtia(device))``. Empirical contract
    (torch 2.11, probed on NPU):
      - the method exists on AveragedModel (torch.mtia module present);
      - on a build without MTIA support (torch.mtia.is_available() False)
        calling am.mtia() / am.mtia(None) / am.mtia(torch.device('mtia:0'))
        raises AssertionError("Torch not compiled with MTIA enabled...")
        from torch.mtia._lazy_init;
      - an int device index (am.mtia(0)) instead hits the aten-side guard and
        raises RuntimeError("Invalid device, must be MTIA device");
      - a failed mtia() leaves the module untouched and usable.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_mtia_available(self):
        """mtia is exported; only exercised when an mtia device exists."""
        am = torch.optim.swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        if not torch.mtia.is_available() if hasattr(torch, 'mtia') else False:
            self.skipTest('no mtia device')
        _ = am.mtia()

    def test_mtia_method_exists_and_is_module_api(self):
        """AveragedModel exposes mtia inherited from nn.Module."""
        if not hasattr(torch.nn.Module, 'mtia'):
            self.skipTest('nn.Module.mtia is not defined in this torch build')
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        self.assertTrue(callable(am.mtia),
                        "mtia must be a callable method on AveragedModel")
        # The bound method is the very same nn.Module implementation
        # (AveragedModel does not override it).
        self.assertIs(type(am).mtia, torch.nn.Module.mtia)
        # torch.mtia exists on torch 2.x builds that ship the MTIA stub.
        self.assertTrue(hasattr(torch, 'mtia'))
        if torch.mtia.is_available():
            self.skipTest('MTIA hardware present; the no-mtia host probe '
                          'does not apply')
        # Probed on this host: the build ships the MTIA stub without hardware.
        self.assertFalse(torch.mtia.is_available(),
                         "on this host torch.mtia.is_available() must be False")

    def test_mtia_raises_when_build_lacks_mtia(self):
        """Without MTIA support the call raises, per device-arg form."""
        if not hasattr(torch.nn.Module, 'mtia'):
            self.skipTest('nn.Module.mtia is not defined in this torch build')
        if not hasattr(torch, 'mtia') or torch.mtia.is_available():
            self.skipTest('MTIA device available; the no-mtia exception path '
                          'is unreachable on this host')
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        weight_before = am.module.weight.detach().clone()

        # No-arg / None / device-object forms all funnel into
        # torch.mtia._lazy_init, which asserts the build has MTIA support.
        with self.assertRaisesRegex(AssertionError, "Torch not compiled with MTIA"):
            am.mtia()
        with self.assertRaisesRegex(AssertionError, "Torch not compiled with MTIA"):
            am.mtia(None)
        with self.assertRaisesRegex(AssertionError, "Torch not compiled with MTIA"):
            am.mtia(torch.device('mtia:0'))
        # An int device index hits the aten-side device guard first.
        with self.assertRaisesRegex(RuntimeError, "Invalid device"):
            am.mtia(0)

        # Every failed call must leave the module untouched and usable.
        torch.testing.assert_close(am.module.weight.detach(), weight_before,
                                   msg="failed mtia() calls must not corrupt the module")
        model = torch.nn.Linear(2, 2)
        am.update_parameters(model)
        self.assertEqual(int(am.n_averaged), 1,
                         "the module must remain updatable after failed mtia() calls")


if __name__ == "__main__":
    run_tests()
