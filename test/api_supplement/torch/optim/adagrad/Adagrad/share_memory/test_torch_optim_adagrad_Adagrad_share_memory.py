# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.adagrad.Adagrad.share_memory 接口功能正确性
API 名称：torch.optim.adagrad.Adagrad.share_memory
API 签名：share_memory(self) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | step 产生 state 后调用；构造期即时初始化的 state 亦可调用 | 已覆盖（_after_step_marks_sum_shared/_before_step_ctor_state） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 无参方法 | 已覆盖（全部用例无参调用） |
| 传参与不传参 | 无参调用 | 已覆盖 |
| 等价类/边界值 | 仅作用于 state['sum']，step 张量保持非共享；state 数值不变 | 已覆盖（_after_step_marks_sum_shared） |
| 正常传参场景 | 返回 None；CPU is_shared=True；设备可调用、state 不变 | 已覆盖（_after_step_marks_sum_shared/_current_device_state_intact） |
| 异常传参场景 | 无参方法，无异常入参路径 | 未覆盖（无该类参数） |
| 混合设备类型 | CPU 驻留参数验证 is_shared；设备驻留参数验证调用与数值完整性 | 已覆盖（_after_step_marks_sum_shared/_current_device_state_intact） |

未覆盖项及原因：
- is_shared() 为 True 的断言仅在 CPU 驻留张量上成立（API 文档语义为进程间共享内存）；
  实测 NPU 上 share_memory() 可调用不报错但 sum.is_shared() 保持 False（CUDA 文档则为恒
  True），故设备驻留路径不作 is_shared 移植性断言，改为验证调用安全与 state 数值不变，
  并在 _current_device_state_intact 内按设备守卫说明。
- step 计数张量（0-dim）不在 share_memory 作用范围（源码文档：仅对 state sum 张量调用
  share_memory_()），该契约在 _after_step_marks_sum_shared 中断言。
"""
import os
import unittest

import torch

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
class TestAdagradshare_memory(TestCase):
    """Test cases for torch.optim.adagrad.Adagrad.share_memory (NPU/CUDA portable).

    Adagrad is the only optimizer family defining share_memory: it calls
    tensor.share_memory_() on every state["sum"] and returns None. Verified
    behaviors locked in here:
      - CPU-resident sum tensors flip is_shared() to True after the call;
      - the 0-dim "step" counter is deliberately NOT shared;
      - the call is safe on accelerator-resident state and preserves values
        (on NPU share_memory_() is accepted but is_shared() stays False, so
        the portable device-path assertions are callability + value identity).
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _step_once(self, opt):
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.ones_like(p)
        opt.step()
        return p

    def test_share_memory_after_step_marks_sum_shared(self):
        """After a step, share_memory marks the CPU-resident sum as shared."""
        # CPU-resident params: the is_shared() contract holds on every host.
        p = torch.nn.Parameter(torch.randn(3, device="cpu"), requires_grad=True)
        opt = torch.optim.Adagrad([p], lr=0.01)
        p = self._step_once(opt)
        self.assertIn("sum", opt.state[p], "step must produce the sum state tensor")
        sum_before = opt.state[p]["sum"].detach().clone()
        result = opt.share_memory()
        self.assertIsNone(result, "share_memory must return None")
        # The documented effect: state["sum"] enters shared memory.
        self.assertTrue(opt.state[p]["sum"].is_shared(),
                        "CPU sum tensor must be shared after share_memory()")
        # Values survive the storage migration to shared memory.
        torch.testing.assert_close(opt.state[p]["sum"], sum_before)
        # Contract from the API docstring: only the "sum" tensors are shared,
        # so the 0-dim step counter must remain non-shared.
        self.assertFalse(opt.state[p]["step"].is_shared(),
                         "share_memory must only touch state['sum'] tensors")

    def test_share_memory_before_step_ctor_state(self):
        """Adagrad eagerly initializes state in the ctor, so share_memory works pre-step."""
        p = torch.nn.Parameter(torch.randn(3, device="cpu"), requires_grad=True)
        opt = torch.optim.Adagrad([p], lr=0.01)
        # No step() yet: Adagrad.__init__ already created step/sum for every
        # param (unlike lazy-init optimizers), so share_memory has state to act on.
        self.assertIn("sum", opt.state[p])
        self.assertIsNone(opt.share_memory())
        self.assertTrue(opt.state[p]["sum"].is_shared(),
                        "ctor-initialized sum must be shareable without any step")

    def test_share_memory_current_device_state_intact(self):
        """share_memory is callable on the active device and preserves state values."""
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adagrad([p], lr=0.01)
        p = self._step_once(opt)
        sum_before = opt.state[p]["sum"].detach().clone()
        step_before = opt.state[p]["step"].detach().clone()
        result = opt.share_memory()
        self.assertIsNone(result, "share_memory must return None")
        # The call must not corrupt the accumulator or the step counter.
        torch.testing.assert_close(opt.state[p]["sum"], sum_before,
                                   msg="share_memory must not change sum values")
        torch.testing.assert_close(opt.state[p]["step"], step_before)
        if self.device_name == 'cpu':
            self.assertTrue(opt.state[p]["sum"].is_shared())
        else:
            # Probed on NPU: share_memory_() is accepted on device tensors but
            # is_shared() does not flip to True (CUDA documents is_shared() as
            # always True, NPU reports False), so no portable is_shared()
            # assertion is possible on the accelerator path; the guaranteed
            # observable contract here is callability plus value identity.
            self.skipTest(
                "is_shared()=True only holds for CPU-resident shared-memory "
                "storage; on accelerator tensors share_memory_() is a no-op "
                "(verified on NPU: is_shared() stays False)")


if __name__ == "__main__":
    run_tests()
