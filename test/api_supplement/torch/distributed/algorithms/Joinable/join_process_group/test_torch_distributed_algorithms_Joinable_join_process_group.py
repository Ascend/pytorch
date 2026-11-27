# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.algorithms.Joinable.join_process_group 接口功能正确性
API 名称：torch.distributed.algorithms.Joinable.join_process_group
API 签名：@property @abstractmethod torch.distributed.algorithms.Joinable.join_process_group(self)

覆盖维度表：
| 覆盖维度       | 说明                                         | 覆盖情况                                              |
|----------------|--------------------------------------------|-----------------------------------------------------|
| 抽象属性约束   | 未实现 join_process_group 的子类禁止实例化   | 已覆盖（test_abstract_property_enforcement）          |
| 默认进程组     | 返回 WORLD 进程组并在 Join 中完成不均衡通信  | 已覆盖（test_ddp_and_custom_world_group）             |
| 结果数值对照   | WORLD 进程组下的聚合结果与期望值一致         | 已覆盖（test_ddp_and_custom_world_group）             |
| 进程组生命周期 | init / destroy_process_group 正常            | 已覆盖（test_ddp_and_custom_world_group）             |
| rendezvous     | FileStore 临时文件（无 MASTER_PORT）         | 已覆盖（test_ddp_and_custom_world_group）             |

未覆盖项及原因：
- 自定义 new_group 与冲突进程组异常：冲突进程组已在 Join 用例覆盖，此处精简为默认进程组核心通信与抽象属性约束。

注意：单进程用例验证抽象属性契约；多卡用例在 NPU/GPU 环境验证 DDP 及自定义
     Joinable 的 join_process_group 返回值及在 Join 中的真实 HCCL/NCCL 通信行为。
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


class _IncompleteJoinable(Joinable):
    def __init__(self):
        super().__init__()

    def join_hook(self, **kwargs):
        return JoinHook()

    @property
    def join_device(self):
        return torch.device("cpu")


class _PGJoinHook(JoinHook):
    def __init__(self, joinable):
        self._joinable = joinable

    def main_hook(self):
        t = torch.zeros(4, device=self._joinable.join_device)
        dist.all_reduce(t, group=self._joinable.join_process_group)
        self._joinable.accumulated.add_(t)


class _PGJoinable(Joinable):
    def __init__(self, device, process_group):
        super().__init__()
        self._device = device
        self._process_group = process_group
        self.accumulated = torch.zeros(4, device=device)

    def join_hook(self, **kwargs):
        return _PGJoinHook(self)

    @property
    def join_device(self):
        return self._device

    @property
    def join_process_group(self):
        return self._process_group

    def step(self, value):
        Join.notify_join_context(self)
        t = torch.full((4,), float(value), device=self._device)
        dist.all_reduce(t, group=self._process_group)
        self.accumulated.add_(t)


def _worker_world_group_join(rank, world_size, device_name):
    local_device = torch.device(f"{device_name}:{rank}")
    pg = dist.group.WORLD
    model = nn.Linear(4, 2, bias=False).to(local_device)
    ddp_model = DDP(model, device_ids=[rank])
    assert ddp_model.join_process_group is pg
    joinable = _PGJoinable(local_device, pg)
    assert joinable.join_process_group is pg
    with Join([joinable]):
        for _ in range(1 if rank == 0 else 3):
            joinable.step(rank + 1)
    # Iter 0: 1+2=3; Iter 1,2: 0+2=2 each -> 7.0.
    torch.testing.assert_close(joinable.accumulated.cpu(), torch.full((4,), 7.0))


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDistributedAlgorithmsJoinableJoinProcessGroup(TestCase):
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

    def test_abstract_property_enforcement(self):
        self.assertIn("join_process_group", Joinable.__abstractmethods__)
        with self.assertRaises(TypeError):
            Joinable()
        with self.assertRaises(TypeError):
            _IncompleteJoinable()

    @skipIfUnsupportMultiNPU(2)
    def test_ddp_and_custom_world_group(self):
        self._run_mp(_worker_world_group_join)


if __name__ == "__main__":
    run_tests()
