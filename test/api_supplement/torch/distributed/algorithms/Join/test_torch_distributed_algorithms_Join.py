# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.algorithms.Join 接口功能正确性
API 名称：torch.distributed.algorithms.Join
API 签名：torch.distributed.algorithms.Join(joinables, enable=True, throw_on_early_termination=False, **kwargs)

覆盖维度表：
| 覆盖维度       | 说明                                         | 覆盖情况                                              |
|----------------|--------------------------------------------|-----------------------------------------------------|
| 多卡集合通信   | world_size=2，HCCL 后端真实通信              | 已覆盖（test_uneven_inputs_ddp）                      |
| 同步/异步分支  | notify_join_context 异步 all_reduce 同步     | 已覆盖（test_uneven_inputs_ddp）                      |
| 结果数值对照   | 不均衡输入训练后各 rank 参数与期望数值一致   | 已覆盖（test_uneven_inputs_ddp）                      |
| 进程组生命周期 | init / destroy_process_group 正常            | 已覆盖（全部多卡用例）                                |
| rendezvous     | FileStore 临时文件（无 MASTER_PORT）         | 已覆盖（全部多卡用例）                                |
| 异常路径       | 空 joinables、冲突 process_group 触发异常    | 已覆盖（test_invalid_args）                           |

未覆盖项及原因：
- enable=False、throw_on_early_termination=True 及 kwargs 透传：精简用例以聚焦核心 DDP 不均衡同步与非法参数路径。

注意：多卡用例在 NPU/GPU 环境运行，通信结果以各 rank 的期望数值为 Golden；
     device_count < 2 时以 self.skipTest 归入「环境不具备」，不伪造对照。
"""
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.algorithms import Join, Joinable, JoinHook
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
    raise AssertionError(f"Expected {exc}")


class _MinimalJoinable(Joinable):
    def __init__(self, device, process_group):
        super().__init__()
        self._device = device
        self._process_group = process_group

    def join_hook(self, **kwargs):
        return JoinHook()

    @property
    def join_device(self):
        return self._device

    @property
    def join_process_group(self):
        return self._process_group


def _worker_uneven_inputs_ddp(rank, world_size, device_name):
    local_device = torch.device(f"{device_name}:{rank}")
    torch.manual_seed(42)
    base_model = nn.Linear(4, 2, bias=False).to(local_device)
    with torch.no_grad():
        base_model.weight.fill_(1.0)
    ddp_model = DDP(base_model, device_ids=[rank])
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    with Join([ddp_model], enable=True):
        for _ in range(2 if rank == 0 else 4):
            optimizer.zero_grad()
            x = torch.ones(1, 4, device=local_device) * float(rank + 1)
            ddp_model(x).sum().backward()
            optimizer.step()
    weight_r0 = ddp_model.module.weight.detach().clone()
    dist.broadcast(weight_r0, src=0)
    torch.testing.assert_close(ddp_model.module.weight.detach().cpu(), weight_r0.cpu())
    # Iter 0,1 (both active): grad=(1+2)/2=1.5 (-0.30); Iter 2,3 (rank 0 shadows 0): grad=1.0 (-0.20) -> 0.50.
    torch.testing.assert_close(ddp_model.module.weight.detach().cpu(), torch.full((2, 4), 0.50), rtol=1e-4, atol=1e-4)


def _worker_invalid_args(rank, world_size, device_name):
    local_device = torch.device(f"{device_name}:{rank}")
    _assert_raises(ValueError, Join, [])
    custom_pg = dist.new_group(ranks=[0, 1])
    _assert_raises(
        ValueError,
        Join,
        [_MinimalJoinable(local_device, dist.group.WORLD), _MinimalJoinable(local_device, custom_pg)],
    )


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDistributedAlgorithmsJoin(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def _run_mp(self, fn, world_size=2, *extra):
        avail = (torch.npu if self.device_name == "npu" else torch.cuda).device_count()
        if avail < world_size:
            self.skipTest(f"Requires at least {world_size} {self.device_name} devices, found {avail}")
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

    @skipIfUnsupportMultiNPU(2)
    def test_uneven_inputs_ddp(self):
        self._run_mp(_worker_uneven_inputs_ddp)

    @skipIfUnsupportMultiNPU(2)
    def test_invalid_args(self):
        self._run_mp(_worker_invalid_args)


if __name__ == "__main__":
    run_tests()
