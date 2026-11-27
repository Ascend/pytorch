# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.default_generator 接口功能正确性
API 名称：torch.default_generator
API 签名：torch.default_generator -> torch.Generator  (默认 CPU 生成器实例)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                                  |
|------------------|--------------------------------------------|-----------------------------------------------------------|
| 对象类型/身份    | 默认生成器为 torch.Generator 实例          | 已覆盖：is_generator_instance                             |
| 枚举选项         | 该对象唯一，无枚举                         | 不适用                                                    |
| 参数类型         | manual_seed/set_state 入参类型校验         | 已覆盖：set_state_non_tensor_raises                       |
| 传参与不传参     | seed() 无参 vs manual_seed(seed) 有参      | 已覆盖：seed_returns_int_and_reseeds、                    |
|                  |                                            |         manual_seed_returns_generator_and_sets_initial_seed|
| 等价类/边界值    | 同种子可复现 / 状态往返                      | 已覆盖：manual_seed_makes_randn_deterministic、           |
|                  |                                            |         get_set_state_roundtrip_reproduces_stream         |
| 正常传参场景     | 作为全局默认 RNG 驱动 torch.randn          | 已覆盖：default_generator_drives_global_rng               |
| 异常传参场景     | set_state 尺寸不符/类型不符抛错            | 已覆盖：set_state_wrong_size_raises、                     |
|                  |                                            |         set_state_non_tensor_raises                       |
| 混合设备类型     | 默认生成器固定为 CPU 生成器                | 已覆盖：device_is_cpu（不涉及异构 Tensor 输入）           |

未覆盖项及原因：
- 枚举选项：torch.default_generator 是单例全局对象，无枚举维度。
- 混合设备类型：该对象为 CPU 默认生成器，无 NPU/CUDA 执行路径；CPU/NPU 数值对照不适用，
  仅断言其设备归属为 CPU。

注：该 API 为全局状态类框架对象（框架类）；用例真实改写全局默认生成器状态，
     故在 setUp/tearDown 中保存并还原 get_state()，避免污染其他用例。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe the device only when the supplement suite actually runs. On the default
# CI collection path the class-level skipUnless fires first, so no torch_npu
# import or device initialization happens here.
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
        DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDefaultGeneratorUt(TestCase):
    """Functional tests for torch.default_generator (framework, stateful)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        # This object is the process-wide default RNG; snapshot and restore its
        # state so cases that reseed it do not leak into other tests.
        self._saved_state = torch.default_generator.get_state().clone()

    def tearDown(self):
        torch.default_generator.set_state(self._saved_state)
        super().tearDown()

    def test_is_generator_instance(self):
        """torch.default_generator is a torch.Generator instance."""
        self.assertIsInstance(torch.default_generator, torch.Generator)

    def test_device_is_cpu(self):
        """The default generator belongs to the CPU device."""
        self.assertEqual(torch.default_generator.device, torch.device("cpu"))

    def test_manual_seed_returns_generator_and_sets_initial_seed(self):
        """manual_seed returns the same generator and records the requested seed."""
        gen = torch.default_generator
        self.assertIs(gen.manual_seed(20240607), gen)
        self.assertEqual(gen.initial_seed(), 20240607)

    def test_manual_seed_makes_randn_deterministic(self):
        """Reseeding yields the identical random stream."""
        gen = torch.default_generator
        gen.manual_seed(1234)
        first = torch.randn(8, generator=gen)
        gen.manual_seed(1234)
        second = torch.randn(8, generator=gen)
        self.assertTrue(torch.equal(first, second))
        gen.manual_seed(4321)
        third = torch.randn(8, generator=gen)
        self.assertFalse(torch.equal(first, third))

    def test_default_generator_drives_global_rng(self):
        """The implicit RNG of torch.randn is this same default generator."""
        gen = torch.default_generator
        gen.manual_seed(777)
        implicit = torch.randn(8)
        gen.manual_seed(777)
        explicit = torch.randn(8, generator=gen)
        self.assertTrue(torch.equal(implicit, explicit))

    def test_seed_returns_int_and_reseeds(self):
        """seed() with no argument reseeds from entropy and returns the used seed."""
        gen = torch.default_generator
        used = gen.seed()
        self.assertIsInstance(used, int)
        self.assertEqual(gen.initial_seed(), used)

    def test_get_state_returns_uint8_cpu_tensor(self):
        """get_state returns a CPU uint8 state tensor."""
        state = torch.default_generator.get_state()
        self.assertIsInstance(state, torch.Tensor)
        self.assertEqual(state.dtype, torch.uint8)
        self.assertEqual(state.device.type, "cpu")

    def test_get_set_state_roundtrip_reproduces_stream(self):
        """Restoring a saved state reproduces the following random stream."""
        gen = torch.default_generator
        snapshot = gen.get_state().clone()
        first = torch.randn(8, generator=gen)
        gen.set_state(snapshot)
        second = torch.randn(8, generator=gen)
        self.assertTrue(torch.equal(first, second))

    def test_set_state_wrong_size_raises(self):
        """A state tensor of the wrong size is rejected with RuntimeError."""
        with self.assertRaises(RuntimeError):
            torch.default_generator.set_state(torch.zeros(3, dtype=torch.uint8))

    def test_set_state_non_tensor_raises(self):
        """A non-tensor argument is rejected with TypeError."""
        with self.assertRaises(TypeError):
            torch.default_generator.set_state("not a generator state")


if __name__ == "__main__":
    run_tests()
