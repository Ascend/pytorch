# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.algorithms.ddp_comm_hooks.quantization_hooks.quantization_pertensor_hook 接口功能正确性
API 名称：torch.distributed.algorithms.ddp_comm_hooks.quantization_hooks.quantization_pertensor_hook
API 签名：quantization_pertensor_hook(process_group, bucket) -> Future[Tensor]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|-------------------------------------|--------------------------------------------------|
| 多卡集合通信   | world_size=2，HCCL 后端真实 DDP 梯度通信 | 已覆盖（test_default_world_pg、test_explicit_process_group） |
| 同步/异步分支  | 钩子返回 Future 链并完成异步 all_gather 聚合 | 已覆盖（test_default_world_pg、test_explicit_process_group） |
| 结果数值对照   | 量化聚合后梯度与两 rank 精确均值误差在容差内 | 已覆盖（test_default_world_pg、test_explicit_process_group） |
| 传参与不传参   | process_group=None（默认组）与显式进程组 | 已覆盖（test_default_world_pg、test_explicit_process_group） |
| 进程组生命周期 | FileStore 临时文件 init/destroy_process_group | 已覆盖（全部多卡用例） |
| 异常路径       | bucket=None、非法 process_group 触发预期异常 | 已覆盖（test_invalid_args_raises） |

未覆盖项及原因：
- 无

注意：多卡用例在 2 卡 NPU/CUDA 环境通过 DDP 注册 quantization_pertensor_hook 运行；
     以各 rank 精确梯度均值为 Golden，结合 8-bit 量化容差断言聚合数值正确性。
"""
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.algorithms.ddp_comm_hooks.quantization_hooks import quantization_pertensor_hook
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


class _DummyBucket:

    def __init__(self, buf):
        self._buf = buf

    def buffer(self):
        return self._buf


def _worker_default_world_pg(rank, world_size, device_name):
    dev = f"{device_name}:{rank}"
    torch.manual_seed(0)
    model = nn.Linear(16, 8, bias=True).to(dev)
    ddp = DDP(model, device_ids=[rank])
    ddp.register_comm_hook(None, quantization_pertensor_hook)

    # Rank 0 input=1.0, rank 1 input=3.0 -> mean weight grad=8.0, bias grad=4.0.
    ddp(torch.full((4, 16), float(1 + 2 * rank), device=dev)).sum().backward()
    torch.testing.assert_close(model.weight.grad, torch.full((8, 16), 8.0, device=dev), rtol=0.05, atol=0.15)
    torch.testing.assert_close(model.bias.grad, torch.full((8,), 4.0, device=dev), rtol=0.05, atol=0.15)


def _worker_explicit_process_group(rank, world_size, device_name):
    dev = f"{device_name}:{rank}"
    sub_pg = dist.new_group(ranks=list(range(world_size)))
    torch.manual_seed(3)
    model = nn.Linear(12, 6, bias=False).to(dev)
    ddp = DDP(model, device_ids=[rank])
    ddp.register_comm_hook(sub_pg, quantization_pertensor_hook)

    base = torch.linspace(-2.0, 2.0, 12, device=dev).unsqueeze(0).repeat(2, 1)
    ddp(base * float(rank + 1)).sum().backward()

    expected = (3.0 * torch.linspace(-2.0, 2.0, 12, device=dev)).unsqueeze(0).repeat(6, 1)
    torch.testing.assert_close(model.weight.grad, expected, rtol=0.05, atol=0.2)


def _worker_invalid_args_raises(rank, world_size, device_name):
    _assert_raises(AttributeError, quantization_pertensor_hook, dist.group.WORLD, None)
    bucket = _DummyBucket(torch.ones(8, device=f"{device_name}:{rank}"))
    _assert_raises(AttributeError, quantization_pertensor_hook, "invalid_pg", bucket)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestQuantizationPertensorHook(TestCase):

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

    @skipIfUnsupportMultiNPU(2)
    def test_default_world_pg(self):
        self._run_mp(_worker_default_world_pg)

    @skipIfUnsupportMultiNPU(2)
    def test_explicit_process_group(self):
        self._run_mp(_worker_explicit_process_group)

    @skipIfUnsupportMultiNPU(2)
    def test_invalid_args_raises(self):
        self._run_mp(_worker_invalid_args_raises)


if __name__ == "__main__":
    run_tests()
