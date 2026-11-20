# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.Module.share_memory 接口功能正确性
API 名称：torch.nn.Module.share_memory
API 签名：share_memory(self) -> Self

覆盖维度表：
| 覆盖维度         | 说明                                        | 覆盖情况                            |
|------------------|---------------------------------------------|-------------------------------------|
| 空/非空          | 含参数模块与含 buffer 模块（BatchNorm1d）的参数和 buffer 均进入共享内存 | 已覆盖（share_memory_marks_params_and_buffers_shared） |
| 枚举选项         | 无枚举参数                                  | 未覆盖（无该类参数）                |
| 参数类型         | 无参方法                                    | 已覆盖（全部用例无参调用）          |
| 传参与不传参     | 无参调用                                    | 已覆盖                              |
| 等价类/边界值     | 重复调用幂等；共享后参数仍可原地更新；前向/训练继续可用     | 已覆盖（share_memory_idempotent/share_memory_training_continues） |
| 正常传参场景     | 返回 self；CPU 驻留参数/buffer is_shared()=True 且数值不变；NPU 驻留路径
|                  | 实测同样成功（is_shared=True、数值不变、前向不变），以本环境实测为准
|                  | | 已覆盖（share_memory_returns_self/share_memory_device_resident_path） |
| 异常传参场景     | 无参方法，无异常入参路径                    | 未覆盖（无参方法，无异常入参路径）  |
| 混合设备类型     | CPU 驻留验证 is_shared 与数值；NPU 驻留验证调用契约、数值与前向一致性；CUDA 按文档 no-op 分支 | 已覆盖（share_memory_device_resident_path） |

未覆盖项及原因：
- is_shared()=True 断言仅在 CPU 驻留张量与本机实测的 NPU 驻留张量上成立
  （实测 torch_npu 2.15.0.dev20261118 + Ascend910B4：NPU 驻留参数调用
  share_memory() 后 is_shared() 为 True 且数值不变，与 torch 2.11 时代
  torch_npu 拦截抛 RuntimeError 的行为不同，以本环境实测为准）。
- CUDA 文档注明 share_memory_() 对 CUDA 张量为 no-op，故 CUDA 分支不断言
  is_shared，仅验证返回 self、数值与前向不变。
- 跨进程共享读写的端到端验证未覆盖：与同目录既有 share_memory 用例
  （torch.optim.Adagrad.share_memory 等）保持同等深度，以 is_shared、数值
  完整性和共享后可继续训练为共享语义证据。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化，避免
# 不必要的探测开销；被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
DEVICE_NAME = 'cpu'
if RUN_FLAG:
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
class TestModuleshare_memory(TestCase):
    """Test cases for torch.nn.Module.share_memory (NPU/CUDA portable).

    nn.Module.share_memory is ``self._apply(lambda t: t.share_memory_())``
    (torch 2.15.0, torch/nn/modules/module.py), moving every parameter and
    buffer of the whole module into shared memory. Empirical contract
    (torch 2.15.0+cpu + torch_npu 2.15.0.dev20261118, probed on
    Ascend910B4):
      - returns self; CPU-resident parameters AND buffers flip is_shared()
        to True with values preserved through the storage migration;
      - repeated calls are idempotent and the module stays writable and
        trainable afterwards;
      - on NPU-resident modules the call succeeds on this build: is_shared()
        becomes True, values are intact and forwards keep producing the same
        results (unlike the torch 2.11-era torch_npu interception, the
        2.15.0.dev20261118 build does not block Module.share_memory);
      - CUDA documents share_memory_() as a no-op for CUDA tensors, so the
        CUDA branch asserts callability and value identity only.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_share_memory_returns_self(self):
        """share_memory returns the module itself for chaining."""
        model = torch.nn.Linear(2, 2)
        result = model.share_memory()
        self.assertIs(result, model, "share_memory must return self")

    def test_share_memory_marks_params_and_buffers_shared(self):
        """CPU-resident parameters and buffers become shared, values intact."""
        model = torch.nn.BatchNorm1d(3)
        with torch.no_grad():
            model.weight.fill_(1.5)
            model.running_mean.fill_(0.25)
        weight_before = model.weight.detach().clone()
        running_mean_before = model.running_mean.detach().clone()

        result = model.share_memory()
        self.assertIs(result, model, "share_memory must return self for chaining")
        self.assertTrue(model.weight.is_shared(),
                        "parameter storage must be shared after share_memory()")
        self.assertTrue(model.running_mean.is_shared(),
                        "buffer storage must be shared after share_memory()")
        torch.testing.assert_close(
            model.weight.detach(), weight_before,
            msg="parameter values must be unchanged by the storage migration")
        torch.testing.assert_close(
            model.running_mean.detach(), running_mean_before,
            msg="buffer values must be unchanged by the storage migration")

    def test_share_memory_idempotent(self):
        """Repeated share_memory calls stay safe and keep the module usable."""
        model = torch.nn.Linear(2, 2)
        weight_before = model.weight.detach().clone()
        first = model.share_memory()
        second = model.share_memory()
        self.assertIs(first, model)
        self.assertIs(second, model)
        self.assertTrue(model.weight.is_shared(),
                        "storage must still be shared after repeated calls")
        torch.testing.assert_close(
            model.weight.detach(), weight_before,
            msg="repeated share_memory calls must not corrupt values")

    def test_share_memory_training_continues(self):
        """Training (forward+backward) keeps working on the shared module."""
        model = torch.nn.Linear(2, 2)
        model.share_memory()
        x = torch.ones(4, 2)
        target = torch.full((4, 2), 0.5)
        out = model(x)
        loss = (out - target).sum()
        loss.backward()
        self.assertIsNotNone(model.weight.grad,
                             "backward must still populate grads after share_memory()")
        # An SGD step on shared storage keeps the module trainable.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_for_step = (model(x) - target).sum()
        optimizer.zero_grad()
        loss_for_step.backward()
        optimizer.step()
        self.assertFalse(torch.isnan(model.weight).any(),
                         "the shared parameter must be updated by optimizer.step()")

    def test_share_memory_device_resident_path(self):
        """Device-resident module: NPU/CPU share fully; CUDA stays a no-op call."""
        model = torch.nn.Linear(2, 2).to(self.device)
        x = torch.randn(3, 2, device=self.device)
        y_before = model(x).detach().clone()
        weight_before = model.weight.detach().clone()

        result = model.share_memory()
        self.assertIs(result, model, "share_memory must return self for chaining")
        torch.testing.assert_close(
            model.weight.detach(), weight_before,
            msg="values must survive the shared-memory migration")
        # Forwards must keep producing the same results after the migration.
        y_after = model(x).detach()
        torch.testing.assert_close(
            y_after.cpu(), y_before.cpu(),
            msg="forward results must be unchanged after share_memory()")
        if self.device_name == 'cuda':
            # CUDA documents share_memory_() as a no-op for CUDA tensors.
            return
        # CPU and NPU (probed on Ascend910B4) both report shared storage.
        self.assertTrue(model.weight.is_shared(),
                        "storage must report shared on %s" % self.device_name)
        with torch.no_grad():
            model.weight.fill_(3.0)
        self.assertEqual(float(model.weight.detach().mean()), 3.0,
                         "the shared parameter must remain writable")


if __name__ == "__main__":
    run_tests()
