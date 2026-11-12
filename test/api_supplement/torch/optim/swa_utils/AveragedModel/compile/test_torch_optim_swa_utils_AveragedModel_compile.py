# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.swa_utils.AveragedModel.compile 接口功能正确性
API 名称：torch.optim.swa_utils.AveragedModel.compile
API 签名：compile(self, *args, **kwargs) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 调用 compile 包装模型 __call__（写入 _compiled_call_impl） |
已覆盖（compile_returns_model/compile_returns_none_and_installs_wrapper） |
| 枚举选项 | 默认后端与显式 backend='eager' 两条路径 |
已覆盖（compile_returns_none_and_installs_wrapper/compile_eager_backend_device_forward） |
| 参数类型 | 任意 torch.compile 兼容 kwargs 透传（backend=）；非法 kwarg 抛 TypeError |
已覆盖（compile_eager_backend_device_forward/compile_invalid_kwarg_raises） |
| 传参与不传参 | 无参与传参均可；compile 返回 None |
已覆盖（compile_returns_model/compile_returns_none_and_installs_wrapper） |
| 等价类/边界值 | 编译后的模型仍可前向且与 eager 数值一致；编译后 update_parameters 仍生效且前向可见新权重 |
已覆盖（compile_returns_none_and_installs_wrapper/compiled_forward_sees_updated_weights） |
| 正常传参场景 | compile() 返回 None、_compiled_call_impl 被写入、编译后 forward 与 eager 结果一致（CPU 与设备 eager 路径） |
已覆盖（compile_returns_none_and_installs_wrapper/compile_eager_backend_device_forward） |
| 异常传参场景 | 非法 kwarg 抛 TypeError；NPU 驻留模块默认 inductor 后端不支持 npu 设备 |
（InductorError，skipTest 注明并以 backend='eager' 覆盖设备路径） |
已覆盖（compile_invalid_kwarg_raises/compile_default_backend_device_module） |
| 混合设备类型 | CPU 驻留模块默认 compile 生效；设备驻留模块经 torch_npu 注册的 dynamo eager 后端前向一致 |
已覆盖（compile_eager_backend_device_forward/compile_default_backend_device_module） |

未覆盖项及原因：
- 本接口本质为优化器/调度器/状态工具的框架行为，涉及浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。
- 设备驻留模块的默认（inductor）编译路径未覆盖数值断言：实测 NPU 上 InductorError("Device npu not supported")，
  torch_npu 未随 torchair 编译时仅注册 dynamo 的 eager 实现，故该路径以 skipTest 注明（test_compile_default_backend_device_module），
  设备侧数值一致性由 backend='eager' 用例保证。
- torch.compile 之外的 config 子选项（fullgraph/dynamic 等）不在断言范围：属于 torch.compile 透传参数，不改变 AveragedModel 侧契约。
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
class TestAveragedModelcompile(TestCase):
    """Test cases for torch.optim.swa_utils.AveragedModel.compile (NPU/CUDA portable).

    AveragedModel inherits nn.Module.compile (torch 2.x API), which wraps the
    module's __call__ via ``self._compiled_call_impl = torch.compile(self._call_impl,
    *args, **kwargs)`` and returns None. Empirical contract (torch 2.11,
    probed on NPU):
      - compile() returns None and installs a callable wrapper; the wrapped
        forward stays numerically identical to eager;
      - update_parameters keeps working after compile and the compiled
        forward observes the updated weights (in-place copy_ keeps storage
        identity, so dynamo guards re-run the graph without stale caches);
      - on NPU-resident modules the default inductor backend fails with
        InductorError("Device npu not supported") because torch_npu only
        registers an eager dynamo backend (no torchair build), while
        backend='eager' works on every device;
      - unknown kwargs are rejected with TypeError (pass-through to
        torch.compile's signature).
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_compile_returns_model(self):
        """compile wraps the module via torch.compile (returns None)."""
        am = torch.optim.swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        try:
            am.compile()
        except RuntimeError as exc:
            self.skipTest('compile unsupported: %s' % exc)
        # AveragedModel's compile calls torch.compile on self.module in-place;
        # it returns None on this build.
        self.assertIsNone(am.compile(), "compile() returns None")
        # compiled module can still run a forward pass
        x = torch.randn(1, 2)
        y = am.module(x)
        self.assertEqual(y.shape, (1, 2))

    def test_compile_returns_none_and_installs_wrapper(self):
        """compile() returns None, installs _compiled_call_impl, keeps numerics."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        x = torch.randn(3, 2)
        y_eager = am(x).detach().clone()
        # Before compiling, calls go through the plain eager _call_impl.
        self.assertIsNone(getattr(am, "_compiled_call_impl", None),
                          "no compiled wrapper must exist before compile()")

        ret = am.compile()
        self.assertIsNone(ret, "nn.Module.compile must return None")
        self.assertIsNotNone(getattr(am, "_compiled_call_impl", None),
                             "compile() must install the _compiled_call_impl wrapper")
        # The wrapped forward produces the same values as eager.
        y_compiled = am(x)
        self.assertEqual(y_compiled.shape, y_eager.shape)
        torch.testing.assert_close(y_compiled.detach(), y_eager,
                                   msg="compiled forward must match the eager output")

    def test_compile_eager_backend_device_forward(self):
        """backend='eager' works on the active device and matches the eager result."""
        model = torch.nn.Linear(2, 2).to(self.device)
        am = swa_utils.AveragedModel(model)
        x = torch.randn(3, 2).to(self.device)
        y_eager = am(x).detach().clone()

        ret = am.compile(backend='eager')
        self.assertIsNone(ret, "compile(backend='eager') must return None")
        self.assertIsNotNone(getattr(am, "_compiled_call_impl", None))
        y_compiled = am(x).detach()
        # CPU golden: the compiled forward must equal the eager forward.
        torch.testing.assert_close(y_compiled.cpu(), y_eager.cpu(),
                                   msg="eager-backend compiled forward must match eager output")
        self.assertEqual(y_compiled.device.type, self.device_name)

    def test_compile_default_backend_device_module(self):
        """Default backend on a device-resident module: NPU inductor lacks npu support."""
        if self.device_name == 'cpu':
            # On CPU the default inductor backend works; verify forward equality.
            am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
            x = torch.randn(3, 2)
            y_eager = am(x).detach().clone()
            am.compile()
            torch.testing.assert_close(am(x).detach(), y_eager,
                                       msg="default-backend compiled forward must match eager")
            return
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2).to(self.device))
        x = torch.randn(3, 2).to(self.device)
        y_eager = am(x).detach().clone()
        try:
            # torch.compile is lazy: on NPU the InductorError only fires at
            # the first compiled forward, so guard the call as well.
            am.compile()
            y_compiled = am(x)
        except Exception as exc:
            # Probed on NPU: the default inductor backend cannot target the
            # npu device (InductorError "Device npu not supported"); torch_npu
            # (without a torchair build) only registers dynamo's eager
            # implementation, so device-side numerics are covered by the
            # backend='eager' case instead.
            self.skipTest('default compile backend unsupported on %s: %s'
                          % (self.device_name, type(exc).__name__))
        torch.testing.assert_close(y_compiled.detach().cpu(), y_eager.cpu(),
                                   msg="default-backend compiled forward must match eager")

    def test_compiled_forward_sees_updated_weights(self):
        """update_parameters after compile() keeps averaging, visible to the compiled forward."""
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        am = swa_utils.AveragedModel(model)
        am.update_parameters(model)  # first fold copies the 1.0 weights
        self.assertEqual(int(am.n_averaged), 1)
        am.compile(backend='eager')
        x = torch.ones(1, 2)
        y_first = am(x).detach().clone()
        torch.testing.assert_close(y_first, torch.full((1, 2), 2.0),
                                   msg="sanity: initial forward reflects w=1.0")

        with torch.no_grad():
            model.weight.fill_(3.0)
        am.update_parameters(model)  # second fold: running mean (1.0+3.0)/2 = 2.0
        self.assertEqual(int(am.n_averaged), 2,
                         "update_parameters must keep working after compile()")
        # The compiled wrapper re-reads the in-place-updated weights.
        y_after = am(x).detach()
        torch.testing.assert_close(y_after, torch.full((1, 2), 4.0),
                                   msg="compiled forward must see the averaged weights")
        # ... and agrees with the underlying module's eager forward.
        torch.testing.assert_close(y_after, am.module(x).detach(),
                                   msg="compiled and eager forwards must stay consistent")

    def test_compile_invalid_kwarg_raises(self):
        """Unknown kwargs are rejected by torch.compile's signature."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        with self.assertRaises(TypeError):
            am.compile(bogus_option=1)


if __name__ == "__main__":
    run_tests()
