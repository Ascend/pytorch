# -*- coding: utf-8 -*-
"""
测试目的：验证 bf16_compress_hook 接口功能正确性
API 名称：torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook
API 签名：bf16_compress_hook(process_group, bucket)

覆盖维度表：
| 覆盖维度       | 说明                                         | 覆盖情况                                              |
|----------------|--------------------------------------------|-----------------------------------------------------|
| 多卡集合通信   | world_size=2，HCCL/NCCL 后端 BF16 压缩聚合   | 已覆盖（test_default_group，NPU 暂跳过）              |
| 进程组参数     | process_group=None（默认 WORLD）             | 已覆盖（test_default_group，NPU 暂跳过）              |
| 压缩与解压语义 | 压缩至 bfloat16 通信后恢复原始 float32 dtype | 已覆盖（test_default_group，NPU 暂跳过）              |
| 结果数值对照   | 聚合后梯度与理论均值及跨 rank 一致性核对     | 已覆盖（test_default_group，NPU 暂跳过）              |
| 进程组生命周期 | init / destroy_process_group 正常            | 已覆盖（全部多卡用例）                                |
| rendezvous     | FileStore 临时文件（无 MASTER_PORT）         | 已覆盖（全部多卡用例）                                |
| 异常路径       | 非法 bucket 参数触发 AttributeError          | 已覆盖（test_invalid_bucket_raises）                  |

未覆盖项及原因：
- NPU 上 DDP._check_comm_hook() 尚未放行 NPU/HCCL 注册 BF16 压缩钩子，test_default_group 在 NPU 跳过。
- 自定义 new_group 与量化边界单测：精简冗余多卡用例，聚焦默认通信路径与非法 bucket 异常。

注意：多卡用例在 NPU/GPU 环境运行，通信梯度以理论均值（按 BF16 表示精度）为 Golden；
     device_count < 2 时以 self.skipTest 归入「环境不具备」，不伪造对照。
"""
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook
from torch.nn.parallel import DistributedDataParallel as DDP

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


def _assert_raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as err:
        return err
    raise AssertionError(f"Expected {exc.__name__}")


def _worker_default_group(rank, world_size, device_name):
    dev = torch.device(f"{device_name}:{rank}")
    model = nn.Linear(4, 2, bias=False).to(dev)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ddp = DDP(model, device_ids=[rank])
    ddp.register_comm_hook(state=None, hook=bf16_compress_hook)
    x_val = 1.0 if rank == 0 else 3.0
    ddp(torch.full((1, 4), x_val, device=dev, dtype=torch.float32)).sum().backward()
    grad = ddp.module.weight.grad
    assert grad is not None and grad.dtype == torch.float32
    torch.testing.assert_close(grad.cpu(), torch.full((2, 4), 2.0, dtype=torch.float32), rtol=1e-2, atol=1e-2)
    grad_r0 = grad.detach().clone()
    dist.broadcast(grad_r0, src=0)
    torch.testing.assert_close(grad.cpu(), grad_r0.cpu())


def _worker_invalid_bucket(rank, world_size, device_name):
    _assert_raises(AttributeError, bf16_compress_hook, dist.group.WORLD, object())


NPU_SKIP_REASON = "PTA/NPU defect: DDP._check_comm_hook does not yet allow BF16 comm hook on NPU/HCCL"


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestBf16CompressHook(TestCase):

    def _run_mp(self, fn, world_size=2, *extra):
        avail = (torch.npu if DEVICE_NAME == "npu" else torch.cuda).device_count()
        if avail < world_size:
            self.skipTest(f"Requires at least {world_size} {DEVICE_NAME} devices, found {avail}")
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_name = f.name
        try:
            mp.spawn(
                _init_dist,
                args=(world_size, fn, BACKEND, DEVICE_NAME, file_name, *extra),
                nprocs=world_size,
                join=True,
            )
        finally:
            if os.path.exists(file_name):
                os.unlink(file_name)

    @unittest.skipIf(DEVICE_NAME == "npu", NPU_SKIP_REASON)
    @skipIfUnsupportMultiNPU(2)
    def test_default_group(self):
        self._run_mp(_worker_default_group)

    @skipIfUnsupportMultiNPU(2)
    def test_invalid_bucket_raises(self):
        self._run_mp(_worker_invalid_bucket)


if __name__ == "__main__":
    run_tests()
