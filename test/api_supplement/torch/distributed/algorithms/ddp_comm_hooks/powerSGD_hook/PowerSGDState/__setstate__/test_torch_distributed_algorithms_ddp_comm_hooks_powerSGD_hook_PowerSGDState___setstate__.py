# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState.__setstate__ 接口功能正确性
API 名称：torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState.__setstate__
API 签名：PowerSGDState.__setstate__(self, state)

覆盖维度表：
| 覆盖维度       | 说明                                     | 覆盖情况                                               |
|----------------|------------------------------------------|--------------------------------------------------------|
| 空/非空        | 空字典与完整/部分状态字典恢复            | 已覆盖（test_direct_setstate_partial_empty_and_mixed_devices） |
| 枚举选项       | 无枚举入参                               | 未覆盖，state 为普通字典映射，无枚举参数               |
| 参数类型       | dict 与非映射类型（None/int）            | 已覆盖（test_invalid_state_and_uninitialized_pg_raises） |
| 传参与不传参   | 显式调用 __setstate__ 与 pickle 往返触发 | 已覆盖（test_pickle_roundtrip_restores_state_and_buffers） |
| 等价类/边界值  | 默认与自定义超参、含张量内存字典及 RNG   | 已覆盖（test_pickle_roundtrip_restores_state_and_buffers） |
| 正常传参场景   | 恢复全部 slot 并将 process_group 设默认  | 已覆盖（test_pickle_roundtrip_restores_state_and_buffers） |
| 异常传参场景   | 未初始化进程组 ValueError、非法 slot 等  | 已覆盖（test_invalid_state_and_uninitialized_pg_raises） |
| 混合设备类型   | error_dict/p_memory_dict 含设备与 CPU 张量| 已覆盖（test_direct_setstate_partial_empty_and_mixed_devices） |

未覆盖项及原因：
- 枚举选项：PowerSGDState.__setstate__ 仅接收 state 字典参数，无枚举型选项。

注意：__setstate__ 内部调用 distributed_c10d._get_default_group() 将 process_group
     重置为默认进程组；单进程用例通过 FileStore + gloo (world_size=1) 初始化默认组，
     并断言所有超参、内部张量缓冲、RNG 状态及默认 process_group 均正确恢复。
"""
from contextlib import contextmanager
import os
import pickle
import tempfile
import unittest

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Set RUN_SUPPLEMENT_API_TESTS=1 to run supplement API tests"
DEVICE_NAME = "cpu"
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
        DEVICE_NAME = "npu" if torch.npu.is_available() else None
    except (ImportError, AttributeError):
        DEVICE_NAME = None
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    from unittest import TestCase
    run_tests = unittest.main


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestPowerSGDStateSetState(TestCase):

    @contextmanager
    def _single_rank_pg(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_name = f.name
        try:
            dist.init_process_group(backend="gloo", rank=0, world_size=1, store=dist.FileStore(file_name, 1))
            yield
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            if os.path.exists(file_name):
                os.unlink(file_name)

    def _assert_attrs(self, obj, expected):
        for name, val in expected.items():
            self.assertEqual(getattr(obj, name), val)

    def test_pickle_roundtrip_restores_state_and_buffers(self):
        with self._single_rank_pg():
            kwargs = dict(
                matrix_approximation_rank=4,
                start_powerSGD_iter=12,
                min_compression_rate=3,
                use_error_feedback=True,
                warm_start=True,
                orthogonalization_epsilon=1e-6,
                compression_stats_logging_frequency=50,
                batch_tensors_with_same_shape=True,
            )
            stats = dict(
                iter=7,
                total_numel_before_compression=1024,
                total_numel_after_compression=256,
                next_stats_report=57,
            )
            orig = PowerSGDState(process_group=None, random_seed=42, **kwargs)
            for k, v in stats.items():
                setattr(orig, k, v)
            _ = orig.rng.randn(4)
            orig.error_dict[0] = torch.tensor([1.5, -2.5, 3.0], dtype=torch.float32)
            orig.p_memory_dict[0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            orig.q_memory_dict[0] = torch.tensor([[0.5, -0.5], [1.0, 0.0]], dtype=torch.float32)

            payload = pickle.dumps(orig)
            expected_next_rng = orig.rng.randn(5).tolist()
            restored = pickle.loads(payload)

            self.assertIs(restored.process_group, dist.group.WORLD)
            self._assert_attrs(restored, {**kwargs, **stats})
            self.assertEqual(restored.compression_stats(), (4.0, 1024, 256))
            self.assertEqual(restored.rng.randn(5).tolist(), expected_next_rng)
            for name in ("error_dict", "p_memory_dict", "q_memory_dict"):
                torch.testing.assert_close(getattr(restored, name)[0], getattr(orig, name)[0])

    def test_direct_setstate_partial_empty_and_mixed_devices(self):
        with self._single_rank_pg():
            state_obj = PowerSGDState(process_group=None, matrix_approximation_rank=2, start_powerSGD_iter=10)
            state_obj.__setstate__({"iter": 99, "matrix_approximation_rank": 8})
            self.assertIs(state_obj.process_group, dist.group.WORLD)
            self._assert_attrs(state_obj, {"iter": 99, "matrix_approximation_rank": 8, "start_powerSGD_iter": 10})

            state_obj.process_group = None
            state_obj.__setstate__({})
            self.assertIs(state_obj.process_group, dist.group.WORLD)
            self.assertEqual(state_obj.iter, 99)

            dev_t = torch.tensor([1.0, 2.0, 3.0], device=DEVICE_NAME)
            cpu_t = torch.tensor([4.0, 5.0, 6.0], device="cpu")
            state_obj.__setstate__({
                "error_dict": {0: dev_t, 1: cpu_t},
                "p_memory_dict": {0: dev_t.clone()},
                "q_memory_dict": {1: cpu_t.clone()},
            })
            self.assertEqual(state_obj.error_dict[0].device.type, DEVICE_NAME)
            self.assertEqual(state_obj.error_dict[1].device.type, "cpu")
            torch.testing.assert_close(state_obj.error_dict[0].cpu(), dev_t.cpu())
            torch.testing.assert_close(state_obj.error_dict[1], cpu_t)
            torch.testing.assert_close(state_obj.p_memory_dict[0].cpu(), dev_t.cpu())
            torch.testing.assert_close(state_obj.q_memory_dict[1], cpu_t)

    def test_invalid_state_and_uninitialized_pg_raises(self):
        self.assertFalse(dist.is_initialized())
        state_obj = PowerSGDState(process_group=None)
        with self.assertRaises(ValueError):
            state_obj.__setstate__(state_obj.__getstate__())

        with self._single_rank_pg():
            with self.assertRaises(AttributeError):
                state_obj.__setstate__({"nonexistent_slot_xyz": 123})
            for bad_state in (None, 42):
                with self.assertRaises(AttributeError):
                    state_obj.__setstate__(bad_state)


if __name__ == "__main__":
    run_tests()
