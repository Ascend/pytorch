# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._symmetric_memory.get 接口功能正确性
API 名称：torch.distributed._symmetric_memory.get
API 签名：torch.distributed._symmetric_memory.get(dst, hdl, peer, offset=0) -> None

覆盖维度表：
| 覆盖维度       | 说明                                     | 覆盖情况                                       |
|----------------|------------------------------------------|------------------------------------------------|
| 多卡集合通信   | world_size=2，HCCL 下跨 rank 单边 RMA 读 | 已覆盖（test_get_default_offset_multinpu）     |
| 同步/异步分支  | 同步单边读取接口，无 async_op 参数       | 未覆盖（接口无 async_op 参数）                 |
| 结果数值对照   | dst 读取结果与对端期望数值精确一致       | 已覆盖（test_get_default_offset_multinpu）     |
| 进程组生命周期 | init / destroy_process_group 正常        | 已覆盖（test_get_default_offset_multinpu）     |
| rendezvous     | FileStore 临时文件（无 MASTER_PORT）     | 已覆盖（test_get_default_offset_multinpu）     |
| 异常路径       | 缺参、非法句柄、非张量、未知关键字报错   | 已覆盖（test_get_invalid_arguments_raise）     |

未覆盖项及原因：
- 同步/异步分支：接口不提供 async_op 参数。
- NPU 环境缺少 libshmem.so，依赖 symm_mem.empty() 的多卡用例在 NPU 跳过。

注意：多卡用例在 NPU/CUDA 环境运行，以对端写入的已知数值为 Golden，
     使用 torch.testing.assert_close 进行浮点对照；device_count < 2 时 skipTest。
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


def _sync(device_name):
    (torch.npu if device_name == "npu" else torch.cuda).synchronize()


def _init_dist(rank, world_size, fn, backend, device_name, file_name, *extra):
    (torch.npu if device_name == "npu" else torch.cuda).set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size, store=dist.FileStore(file_name, world_size))
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    try:
        fn(rank, world_size, device_name, *extra)
    finally:
        dist.destroy_process_group()


def _worker_get_default_offset(rank, world_size, device_name):
    local_device = f"{device_name}:{rank}"
    buf = symm_mem.empty((16,), dtype=torch.float32, device=local_device)
    buf.copy_(torch.arange(16, dtype=torch.float32, device=local_device) + float((rank + 1) * 100))
    hdl = symm_mem.rendezvous(buf, group=dist.group.WORLD)
    _sync(device_name)
    dist.barrier()

    peer = (rank + 1) % world_size
    dst = torch.zeros(16, dtype=torch.float32, device=local_device)
    symm_mem.get(dst, hdl, peer)
    _sync(device_name)
    dist.barrier()
    torch.testing.assert_close(dst.cpu(), torch.arange(16, dtype=torch.float32) + float((peer + 1) * 100))


NPU_SKIP_REASON = "NPU env missing libshmem.so required by symmetric_memory.empty()"


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestSymmMemGet(TestCase):

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

    @unittest.skipIf(DEVICE_NAME == "npu", NPU_SKIP_REASON)
    @skipIfUnsupportMultiNPU(2)
    def test_get_default_offset_multinpu(self):
        self._run_mp(_worker_get_default_offset)

    def test_get_invalid_arguments_raise(self):
        self.assertIn(self.device_name, ("npu", "cuda"))
        dst = torch.zeros(4, dtype=torch.float32, device=f"{self.device_name}:0")
        cases = [
            (TypeError, (), {}),
            (TypeError, (dst,), {}),
            (TypeError, (dst, None), {}),
            ((TypeError, AttributeError, RuntimeError, ValueError), (dst, None, 0), {}),
            ((TypeError, AttributeError, RuntimeError, ValueError), ("not_a_tensor", None, 0), {}),
            (TypeError, (dst, None, 0), {"unknown_kw": 1}),
        ]
        for exc, args, kw in cases:
            with self.assertRaises(exc):
                symm_mem.get(*args, **kw)


if __name__ == "__main__":
    run_tests()
