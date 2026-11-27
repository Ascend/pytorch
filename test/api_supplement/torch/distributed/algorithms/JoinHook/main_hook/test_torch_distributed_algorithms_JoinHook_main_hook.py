# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.algorithms.JoinHook.main_hook 接口功能正确性
API 名称：torch.distributed.algorithms.JoinHook.main_hook
API 签名：torch.distributed.algorithms.JoinHook.main_hook(self)

覆盖维度表：
| 覆盖维度       | 说明                                         | 覆盖情况                                              |
|----------------|--------------------------------------------|-----------------------------------------------------|
| 默认行为       | 基类 main_hook 默认无操作并返回 None         | 已覆盖（test_default_and_invalid_args）               |
| 子类重写       | 自定义 main_hook 修改状态并执行影子通信      | 已覆盖（test_uneven_ranks_shadow_allreduce）          |
| 多卡不均衡场景 | 提前结束 rank 按差值次数触发 main_hook       | 已覆盖（test_uneven_ranks_shadow_allreduce）          |
| 进程组生命周期 | init / destroy_process_group 正常            | 已覆盖（test_uneven_ranks_shadow_allreduce）          |
| rendezvous     | FileStore 临时文件（无 MASTER_PORT）         | 已覆盖（test_uneven_ranks_shadow_allreduce）          |
| 异常路径       | 传入多余参数触发 TypeError                   | 已覆盖（test_default_and_invalid_args）               |

未覆盖项及原因：
- 多卡均衡与多 Joinable 顺序场景：精简用例以聚焦核心不均衡影子通信与基类/异常参数验证。

注意：单进程用例验证基类方法签名与默认行为；多卡用例在 NPU/GPU 环境通过 Join
     验证 main_hook 在不均衡场景下的调用次数与影子通信数值。
"""
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms import Join, Joinable, JoinHook

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


class _ShadowAllReduceHook(JoinHook):
    def __init__(self, joinable, shadow_val):
        self._joinable = joinable
        self._shadow_val = shadow_val

    def main_hook(self):
        self._joinable.main_hook_calls += 1
        t = torch.full((4,), float(self._shadow_val), device=self._joinable.join_device)
        dist.all_reduce(t, group=self._joinable.join_process_group)
        self._joinable.total.add_(t)


class _ShadowJoinable(Joinable):
    def __init__(self, device, process_group, shadow_val=0.0):
        super().__init__()
        self._device = device
        self._process_group = process_group
        self._shadow_val = shadow_val
        self.main_hook_calls = 0
        self.total = torch.zeros(4, device=device)

    def join_hook(self, **kwargs):
        return _ShadowAllReduceHook(self, self._shadow_val)

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
        self.total.add_(t)


def _worker_uneven_ranks_shadow_allreduce(rank, world_size, device_name):
    joinable = _ShadowJoinable(torch.device(f"{device_name}:{rank}"), dist.group.WORLD, shadow_val=4.0)
    with Join([joinable]):
        for _ in range(1 if rank == 0 else 4):
            joinable.step(rank + 1)
    expected_calls = 3 if rank == 0 else 0
    assert (
        joinable.main_hook_calls == expected_calls
    ), f"Expected {expected_calls} calls, got {joinable.main_hook_calls}"
    # Iter 0: 1+2=3; Iters 1..3: 4+2=6 each -> total 21.
    torch.testing.assert_close(joinable.total.cpu(), torch.full((4,), 21.0))


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDistributedAlgorithmsJoinHookMainHook(TestCase):
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

    def test_default_and_invalid_args(self):
        self.assertIsNone(JoinHook().main_hook())
        with self.assertRaises(TypeError):
            JoinHook().main_hook(1)

    @skipIfUnsupportMultiNPU(2)
    def test_uneven_ranks_shadow_allreduce(self):
        self._run_mp(_worker_uneven_ranks_shadow_allreduce)


if __name__ == "__main__":
    run_tests()
