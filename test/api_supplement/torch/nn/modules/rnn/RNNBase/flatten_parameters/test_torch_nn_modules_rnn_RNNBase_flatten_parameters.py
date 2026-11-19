# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.rnn.RNNBase.flatten_parameters 接口功能正确性
API 名称：torch.nn.modules.rnn.RNNBase.flatten_parameters
API 签名：flatten_parameters(self) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 部分权重为 None（未完全实例化）时静默返回不报错 | 已覆盖（test_flatten_parameters_partial_weights_returns_silently） |
| 枚举选项 | 非 CUDA 权重、dtype 混合、设备混合、别名权重四类分支 | 已覆盖（test_flatten_parameters_noop_cpu_module、
test_flatten_parameters_noop_device_module、test_flatten_parameters_mixed_dtype_returns_silently、
test_flatten_parameters_mixed_device_returns_silently、test_flatten_parameters_aliased_weights_returns_silently） |
| 参数类型 | 无入参方法；Parameter 与非 Tensor 权重分支分别覆盖 | 已覆盖（test_flatten_parameters_partial_weights_returns_silently） |
| 传参与不传参 | 显式调用、构造期 _init_flat_weights 自动调用、设备驻留极小模块（bias=False） | 已覆盖（test_flatten_parameters_noop_cpu_module、
test_flatten_parameters_boundary_minimal_module） |
| 等价类/边界值 | 设备驻留 bias-free 极小模块（2 权重）、CPU 模块、.to() 迁移后模块上重复调用 | 已覆盖（test_flatten_parameters_boundary_minimal_module、
test_flatten_parameters_memory_repeated_calls、test_flatten_parameters_after_to_device） |
| 正常传参场景 | 按设备分支断言真实契约：非 CUDA（NPU/CPU）为 no-op（data_ptr 与权重值不变）；
CUDA 上真实执行参数重排（全部 _flat_weights 共享同一底层存储）且权重值逐位精确不变；
调用前后设备前向输出不变，且与 CPU Golden 在跨后端容差内一致 | 已覆盖（test_flatten_parameters_noop_device_module、
test_flatten_parameters_forward_unchanged_after_call、test_flatten_parameters_after_to_device） |
| 异常传参场景 | 方法无入参校验路径；各早退分支均不得抛异常（静默返回为契约） | 已覆盖（partial、mixed_dtype、mixed_device、aliased 四条早退分支） |
| 混合设备类型 | 权重分属 CPU 与设备时静默返回且权重不受影响 | 已覆盖（test_flatten_parameters_mixed_device_returns_silently） |

未覆盖项及原因：
- 别名权重早退分支（unique_data_ptrs 检查）位于 is_cuda 检查之后；CPU/NPU/GPU 轮验证调用无
  异常且权重不变，CUDA 上别名模块应跳过重排的行为未单独断言。
- GPU attempt-1（run 44296458706）证实 CUDA 上真实重排，据此引入设备分支断言；GPU attempt-2
  （run 44303441707）暴露 boundary 用例构造未传 device 的用例缺陷（CPU 模块被断言 CUDA 契约，
  2 != 1），flatten-r3 修复为设备驻留 bias-free 模块；CUDA bias=False（num_weights=2）的单一
  存储契约首次由本修复真实执行（源码依据：_cudnn_rnn_flatten_weight 将全部 _flat_weights 统一
  重排进 weight_buf 并按 view set_ 回各权重）。

注意：flatten_parameters 无浮点输出。no-op 分支的 Golden 采用权重快照（rtol=0, atol=0 精确对照）；
     CUDA 重排分支断言全部权重共享单一底层存储且逐位精确值不变；前向对照中设备内前后两次前向
     保持不变性断言，设备前向与 CPU Golden 使用跨后端 float32 容差 rtol=1e-2、atol=1e-3
     （依据 GPU attempt-1 实测最大偏差 abs≈1.3e-4、rel≈4.2e-3）。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化；
# 被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
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

# ORDER CONSTRAINT: import torch_npu.testing.testcase BEFORE
# torch_npu.contrib.transfer_to_npu (patching first breaks the import).
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401

from torch.nn.modules.rnn import RNNBase


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRNNBaseFlattenParameters(TestCase):
    """Test cases for torch.nn.modules.rnn.RNNBase.flatten_parameters.

    Device branches follow the documented contract: on non-CUDA devices
    (NPU/CPU) the call is a no-op (storage pointers and values untouched);
    on CUDA the call really repacks the flat weights into one shared storage
    while keeping every weight value bit-exact (GPU attempt-1 evidence).
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _snapshot(self, module):
        """Return [(name, data_ptr, cpu_value)] for every non-None flat weight."""
        return [
            (name, weight.data_ptr(), weight.detach().cpu().clone())
            for name, weight in zip(module._flat_weights_names, module._flat_weights)
            if isinstance(weight, torch.Tensor)
        ]

    def _assert_noop(self, module, snapshot):
        """Non-CUDA contract: storage pointers and exact values are preserved."""
        for name, ptr, value in snapshot:
            weight = getattr(module, name)
            self.assertEqual(weight.data_ptr(), ptr, f"data_ptr changed for {name}")
            torch.testing.assert_close(weight.detach().cpu(), value, rtol=0.0, atol=0.0)

    def _assert_cuda_flattened(self, module, snapshot):
        """CUDA contract: one shared storage, every weight value bit-exact."""
        storages = {w.untyped_storage().data_ptr() for w in module._flat_weights}
        self.assertEqual(
            len(storages),
            1,
            "expected all flat weights to share one storage after flatten_parameters",
        )
        for name, _, value in snapshot:
            weight = getattr(module, name)
            torch.testing.assert_close(weight.detach().cpu(), value, rtol=0.0, atol=0.0)

    def test_flatten_parameters_noop_cpu_module(self):
        """On CPU the documented no-op path keeps every weight byte-identical."""
        module = RNNBase("LSTM", 6, 8)
        snapshot = self._snapshot(module)
        module.flatten_parameters()
        self._assert_noop(module, snapshot)

    def test_flatten_parameters_noop_device_module(self):
        """Device module: no-op on NPU/CPU; single-storage repack on CUDA."""
        module = RNNBase("LSTM", 6, 8, device=self.device_name)
        for weight in module._flat_weights:
            self.assertEqual(weight.device.type, self.device_name)
        snapshot = self._snapshot(module)
        module.flatten_parameters()
        if self.device_name == 'cuda':
            self._assert_cuda_flattened(module, snapshot)
        else:
            self._assert_noop(module, snapshot)

    def test_flatten_parameters_partial_weights_returns_silently(self):
        """A None entry in _flat_weights short-circuits the call without error."""
        module = RNNBase("LSTM", 6, 8)
        module.weight_hh_l0 = None
        self.assertIn(None, module._flat_weights)
        snapshot = self._snapshot(module)
        module.flatten_parameters()
        self._assert_noop(module, snapshot)

    def test_flatten_parameters_mixed_dtype_returns_silently(self):
        """Weights with mixed dtypes short-circuit the call without error."""
        module = RNNBase("GRU", 6, 8)
        module.weight_ih_l0 = torch.nn.Parameter(torch.zeros(24, 6, dtype=torch.float64))
        snapshot = self._snapshot(module)
        module.flatten_parameters()
        self._assert_noop(module, snapshot)

    def test_flatten_parameters_mixed_device_returns_silently(self):
        """Weights on different devices short-circuit the call without error."""
        if self.device_name == 'cpu':
            self.skipTest("requires NPU or CUDA device for mixed-device weights")
        module = RNNBase("GRU", 6, 8)
        module.weight_ih_l0 = torch.nn.Parameter(
            torch.zeros(24, 6, device=self.device_name)
        )
        devices = {w.device.type for w in module._flat_weights}
        self.assertEqual(devices, {"cpu", self.device_name})
        snapshot = self._snapshot(module)
        module.flatten_parameters()
        # The CPU weight fails the is_cuda gate, so no repack happens anywhere.
        self._assert_noop(module, snapshot)

    def test_flatten_parameters_aliased_weights_returns_silently(self):
        """Aliased weights are tolerated: the call must not raise or mutate weights."""
        module = RNNBase("GRU", 6, 8)
        module.weight_hh_l0 = module.weight_ih_l0
        snapshot = self._snapshot(module)
        module.flatten_parameters()
        self._assert_noop(module, snapshot)

    def test_flatten_parameters_boundary_minimal_module(self):
        """A minimal device-resident bias-free module (2 weights) flattens."""
        module = RNNBase("LSTM", 6, 8, bias=False, device=self.device_name)
        self.assertEqual(len(module._flat_weights), 2)
        snapshot = self._snapshot(module)
        module.flatten_parameters()
        if self.device_name == 'cuda':
            self._assert_cuda_flattened(module, snapshot)
        else:
            self._assert_noop(module, snapshot)

    def test_flatten_parameters_after_to_device(self):
        """After .to(device): values preserved; storage contract per device branch."""
        if self.device_name == 'cpu':
            self.skipTest("requires NPU or CUDA device for migration semantics")
        source = RNNBase("LSTM", 6, 8)
        cpu_values = [
            param.detach().cpu().clone() for _, param in source.named_parameters()
        ]
        module = source.to(self.device_name)
        self.assertEqual(module._flat_weights[0].device.type, self.device_name)
        ptrs_before = [w.data_ptr() for w in module._flat_weights]
        module.flatten_parameters()
        if self.device_name == 'cuda':
            snapshot = list(
                zip(module._flat_weights_names, [0] * len(cpu_values), cpu_values)
            )
            self._assert_cuda_flattened(module, snapshot)
        else:
            ptrs_after = [w.data_ptr() for w in module._flat_weights]
            self.assertEqual(ptrs_before, ptrs_after)
            for (_, param), value in zip(module.named_parameters(), cpu_values):
                torch.testing.assert_close(param.detach().cpu(), value, rtol=0.0, atol=0.0)

    def test_flatten_parameters_forward_unchanged_after_call(self):
        """flatten_parameters must not alter a real device forward pass."""
        if self.device_name == 'cpu':
            self.skipTest("requires NPU or CUDA device for forward comparison")
        torch.manual_seed(7)
        cpu_module = torch.nn.LSTM(4, 6, batch_first=True)
        device_module = torch.nn.LSTM(4, 6, batch_first=True, device=self.device_name)
        device_module.load_state_dict(
            {k: v.clone() for k, v in cpu_module.state_dict().items()}
        )
        inputs = torch.randn(2, 3, 4)
        device_inputs = inputs.to(self.device_name)
        with torch.no_grad():
            cpu_out, _ = cpu_module(inputs)
            out_before, _ = device_module(device_inputs)
            device_module.flatten_parameters()
            out_after, _ = device_module(device_inputs)
        # Same device, same weights, same inputs: invariant across the call.
        torch.testing.assert_close(out_after.cpu(), out_before.cpu())
        # Cross-backend CPU Golden with float32 tolerance: GPU attempt-1
        # (run 44296458706, RTX 5070 Ti, torch 2.15.0+cu132) measured a max
        # abs diff of ~1.3e-4 and a max rel diff of ~4.2e-3 against CPU for
        # this LSTM, so the strict default tolerance is not a portable
        # cross-backend contract.
        torch.testing.assert_close(out_before.cpu(), cpu_out, rtol=1e-2, atol=1e-3)

    def test_flatten_parameters_memory_repeated_calls(self):
        """10 repeated calls keep the device contract stable and memory flat."""
        if self.device_name == 'cpu':
            self.skipTest("requires a device allocator (NPU/CUDA)")
        module = RNNBase("LSTM", 6, 8, device=self.device_name)
        snapshot = self._snapshot(module)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        for _ in range(10):
            module.flatten_parameters()
            if self.device_name == 'cuda':
                storages = {w.untyped_storage().data_ptr() for w in module._flat_weights}
                self.assertEqual(
                    len(storages),
                    1,
                    "repeated calls must keep the packed single storage",
                )
        if self.device_name == 'cuda':
            self._assert_cuda_flattened(module, snapshot)
        else:
            self._assert_noop(module, snapshot)
        torch.cuda.empty_cache()
        grown = torch.cuda.memory_allocated()
        self.assertLessEqual(grown - base, 1024 * 1024)


if __name__ == "__main__":
    run_tests()
