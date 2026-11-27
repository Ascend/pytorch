# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._symmetric_memory.is_symm_mem_tensor 接口功能正确性
API 名称：torch.distributed._symmetric_memory.is_symm_mem_tensor
API 签名：torch.distributed._symmetric_memory.is_symm_mem_tensor(tensor) -> bool

覆盖维度表：
| 覆盖维度       | 说明                                      | 覆盖情况                                               |
|----------------|-------------------------------------------|--------------------------------------------------------|
| 多卡集合通信   | world_size=2，HCCL 下对称内存分配与判定   | 已覆盖（test_symm_mem_tensor_returns_true_multinpu）   |
| 同步/异步分支  | 非集合 O(1) 查询，无异步分支              | 未覆盖（接口为同步布尔查询，无 async_op 参数）         |
| 结果数值对照   | 普通张量返回 False，对称内存张量返回 True | 已覆盖（test_regular_device_tensors_return_false、多卡）|
| 进程组生命周期 | init / destroy_process_group 正常         | 已覆盖（test_symm_mem_tensor_returns_true_multinpu）   |
| rendezvous     | FileStore 临时文件（无 MASTER_PORT）      | 已覆盖（test_symm_mem_tensor_returns_true_multinpu）   |
| 异常路径       | 缺参、None、非 Tensor、未知关键字报异常   | 已覆盖（test_invalid_arguments_raise）                 |

未覆盖项及原因：
- 同步/异步分支：接口为非集合 O(1) 同步查询，无 async_op 参数。
- NPU 环境缺少 libshmem.so，test_symm_mem_tensor_returns_true_multinpu 在 NPU 跳过。

注意：多卡用例在 NPU/CUDA 环境运行，验证 symm_mem.empty 分配及 rendezvous 后返回 True，
     普通设备张量与 CPU 张量返回 False；device_count < 2 时以 skipTest 归入环境不具备。
"""
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp

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
BACKEND = "hccl" if DEVICE_NAME == "npu" else "nccl"
try:
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
except (ImportError, TypeError):
    from unittest import TestCase
    run_tests = unittest.main

    def skipIfUnsupportMultiNPU(n):
        return lambda fn: fn


def _init_dist(rank, world_size, fn, backend, device_name, file_name, *extra):
    (torch.npu if device_name == "npu" else torch.cuda).set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size, store=dist.FileStore(file_name, world_size))
    try:
        fn(rank, world_size, device_name, *extra)
    finally:
        dist.destroy_process_group()


def _worker_check_symm_mem_tensor(rank, world_size, device_name):
    local_device = f"{device_name}:{rank}"
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    regular = torch.zeros(16, dtype=torch.float32, device=local_device)
    assert symm_mem.is_symm_mem_tensor(regular) is False, f"Expected False for regular tensor on rank {rank}"

    symm_tensor = symm_mem.empty((16,), dtype=torch.float32, device=local_device)
    assert symm_mem.is_symm_mem_tensor(symm_tensor) is True, f"Expected True before rendezvous on rank {rank}"

    symm_mem.rendezvous(symm_tensor, group=dist.group.WORLD)
    assert symm_mem.is_symm_mem_tensor(symm_tensor) is True, f"Expected True after rendezvous on rank {rank}"
    dist.barrier()


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestIsSymmMemTensor(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def _run_mp(self, fn, world_size=2, *extra):
        count = (torch.npu if self.device_name == "npu" else torch.cuda).device_count()
        if count < world_size:
            self.skipTest(f"Requires at least {world_size} {self.device_name} devices, got {count}")
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_name = f.name
        try:
            mp.spawn(
                _init_dist,
                args=(world_size, fn, BACKEND, self.device_name, file_name, *extra),
                nprocs=world_size,
                join=True,
            )
        finally:
            if os.path.exists(file_name):
                os.unlink(file_name)

    def test_regular_device_tensors_return_false(self):
        self.assertIn(self.device_name, ("npu", "cuda"))
        dev = f"{self.device_name}:0"
        cpu_tensor = torch.zeros(4, dtype=torch.float32, device="cpu")
        tensors = [
            torch.zeros(4, dtype=torch.float32, device=dev),
            torch.empty((0,), dtype=torch.float32, device=dev),
            torch.tensor(1.0, dtype=torch.float16, device=dev),
            torch.ones((2, 3), dtype=torch.bfloat16, device=dev),
            torch.arange(6, dtype=torch.int64, device=dev).reshape(2, 3).t(),
            cpu_tensor,
        ]
        for tensor in tensors:
            res = symm_mem.is_symm_mem_tensor(tensor)
            self.assertIsInstance(res, bool)
            self.assertIs(res, False)
        self.assertIs(symm_mem.is_symm_mem_tensor(tensor=cpu_tensor), False)

    def test_invalid_arguments_raise(self):
        cases = [
            ((), {}),
            ((None,), {}),
            (([1, 2, 3],), {}),
            (("not_a_tensor",), {}),
            ((), {"x": torch.zeros(2)}),
        ]
        for args, kw in cases:
            with self.assertRaises(TypeError):
                symm_mem.is_symm_mem_tensor(*args, **kw)

    @unittest.skipIf(DEVICE_NAME == "npu", "NPU env missing libshmem.so required by symmetric_memory.empty()")
    @skipIfUnsupportMultiNPU(2)
    def test_symm_mem_tensor_returns_true_multinpu(self):
        self._run_mp(_worker_check_symm_mem_tensor)


if __name__ == "__main__":
    run_tests()
