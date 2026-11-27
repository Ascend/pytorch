# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.algorithms.JoinHook 接口功能正确性
API 名称：torch.distributed.algorithms.JoinHook
API 签名：torch.distributed.algorithms.JoinHook()

覆盖维度表：
| 覆盖维度       | 说明                                         | 覆盖情况                                              |
|----------------|--------------------------------------------|-----------------------------------------------------|
| 默认基类行为   | 实例化及默认 main_hook/post_hook 返回 None   | 已覆盖（test_default_and_invalid_arguments）          |
| 子类重写       | 继承 JoinHook 重写 main_hook 与 post_hook    | 已覆盖（test_join_integration_multinpu）              |
| 多卡集合通信   | world_size=2，HCCL 后端真实 Join 集成        | 已覆盖（test_join_integration_multinpu）              |
| 结果数值对照   | 影子 all_reduce 数值及 is_last_joiner 标志   | 已覆盖（test_join_integration_multinpu）              |
| 进程组生命周期 | init / destroy_process_group 正常            | 已覆盖（test_join_integration_multinpu）              |
| rendezvous     | FileStore 临时文件（无 MASTER_PORT）         | 已覆盖（test_join_integration_multinpu）              |
| 异常路径       | 多余构造参数或非法参数个数触发 TypeError     | 已覆盖（test_default_and_invalid_arguments）          |

未覆盖项及原因：
- 无

注意：单进程用例验证基类契约与异常参数；多卡用例在 NPU/GPU 环境验证 JoinHook
     在 Join 上下文中的真实影子通信与 post_hook 回调语义，以期望数值为 Golden。
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


class _RecordingJoinHook(JoinHook):
    def __init__(self, device=None, process_group=None):
        self.device = device
        self.process_group = process_group
        self.events = []
        self.shadow_sum = torch.zeros(4, device=device) if device is not None else None

    def main_hook(self):
        self.events.append("main")
        if self.device is not None and self.process_group is not None:
            t = torch.full((4,), 3.0, device=self.device)
            dist.all_reduce(t, group=self.process_group)
            self.shadow_sum.add_(t)

    def post_hook(self, is_last_joiner):
        self.events.append(("post", is_last_joiner))


class _HookTestJoinable(Joinable):
    def __init__(self, device, process_group):
        super().__init__()
        self._device = device
        self._process_group = process_group
        self.hook = _RecordingJoinHook(device=device, process_group=process_group)
        self.active_sum = torch.zeros(4, device=device)

    def join_hook(self, **kwargs):
        return self.hook

    @property
    def join_device(self):
        return self._device

    @property
    def join_process_group(self):
        return self._process_group

    def run_step(self, value):
        Join.notify_join_context(self)
        t = torch.full((4,), float(value), device=self._device)
        dist.all_reduce(t, group=self._process_group)
        self.active_sum.add_(t)


def _worker_join_integration(rank, world_size, device_name):
    joinable = _HookTestJoinable(torch.device(f"{device_name}:{rank}"), dist.group.WORLD)
    with Join([joinable]):
        for _ in range(1 if rank == 0 else 3):
            joinable.run_step(rank + 1)
    # Iter 0: 1+2=3; Iter 1,2: 3(shadow)+2(active)=5 each -> total 13.
    torch.testing.assert_close((joinable.active_sum + joinable.hook.shadow_sum).cpu(), torch.full((4,), 13.0))
    expected_events = ["main", "main", ("post", False)] if rank == 0 else [("post", True)]
    assert joinable.hook.events == expected_events, f"Unexpected events on rank {rank}: {joinable.hook.events}"


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDistributedAlgorithmsJoinHook(TestCase):
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

    def test_default_and_invalid_arguments(self):
        hook = JoinHook()
        self.assertIsInstance(hook, JoinHook)
        self.assertIsNone(hook.main_hook())
        self.assertIsNone(hook.post_hook(True))
        self.assertIsNone(hook.post_hook(False))
        with self.assertRaises(TypeError):
            JoinHook("unexpected_arg")
        with self.assertRaises(TypeError):
            hook.main_hook("extra")
        with self.assertRaises(TypeError):
            hook.post_hook()

    @skipIfUnsupportMultiNPU(2)
    def test_join_integration_multinpu(self):
        self._run_mp(_worker_join_integration)


if __name__ == "__main__":
    run_tests()
