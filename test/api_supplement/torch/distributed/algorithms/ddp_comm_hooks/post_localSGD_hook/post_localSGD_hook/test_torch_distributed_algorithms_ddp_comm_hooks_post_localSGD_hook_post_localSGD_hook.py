# -*- coding: utf-8 -*-
"""
测试目的：验证 post_localSGD_hook 接口功能正确性
API 名称：torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook.post_localSGD_hook
API 签名：post_localSGD_hook(state, bucket)

覆盖维度表：
| 覆盖维度       | 说明                                         | 覆盖情况                                              |
|----------------|--------------------------------------------|-----------------------------------------------------|
| 多卡集合通信   | world_size=2，HCCL 后端全局梯度聚合与切换    | 已覆盖（test_no_local_allreduce）                     |
| 迭代阶段切换   | iter < start_localSGD_iter 与 >= 切换行为    | 已覆盖（test_no_local_allreduce）                     |
| 开关分支       | post_local_gradient_allreduce=False          | 已覆盖（test_no_local_allreduce）                     |
| 结果数值对照   | 切换前后梯度分别为全局均值与局部梯度         | 已覆盖（test_no_local_allreduce）                     |
| 进程组生命周期 | init / destroy_process_group 正常            | 已覆盖（test_no_local_allreduce）                     |
| rendezvous     | FileStore 临时文件（无 MASTER_PORT）         | 已覆盖（test_no_local_allreduce）                     |
| 异常路径       | 非法 state 触发 AttributeError               | 已覆盖（test_invalid_state_raises）                   |

未覆盖项及原因：
- 显式 subgroup 与 subgroup=None 懒创建分支：精简冗余多卡用例，聚焦核心阶段切换与非法 state 异常。

注意：多卡用例在 NPU/GPU 环境运行，通过对比 start_localSGD_iter 前后各 rank 梯度数值，
     验证全局聚合与跳过局部聚合核心路径。
"""
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
    PostLocalSGDState,
    post_localSGD_hook,
)
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


def _worker_no_local_allreduce(rank, world_size, device_name):
    state = PostLocalSGDState(
        process_group=None, subgroup=None, start_localSGD_iter=1, post_local_gradient_allreduce=False
    )
    dev = torch.device(f"{device_name}:{rank}")
    model = nn.Linear(4, 2, bias=False).to(dev)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ddp = DDP(model, device_ids=[rank])
    ddp.register_comm_hook(state=state, hook=post_localSGD_hook)
    x = torch.full((1, 4), 1.0 if rank == 0 else 5.0, device=dev)

    # Iter 0 (< start_localSGD_iter): global allreduce -> 3.0; Iter 1: local grad kept.
    ddp.zero_grad()
    ddp(x).sum().backward()
    torch.testing.assert_close(ddp.module.weight.grad.cpu(), torch.full((2, 4), 3.0), rtol=1e-4, atol=1e-4)
    assert state.iter == 1
    ddp.zero_grad()
    ddp(x).sum().backward()
    torch.testing.assert_close(
        ddp.module.weight.grad.cpu(), torch.full((2, 4), 1.0 if rank == 0 else 5.0), rtol=1e-4, atol=1e-4
    )


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestPostLocalSGDHook(TestCase):

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
    def test_no_local_allreduce(self):
        self._run_mp(_worker_no_local_allreduce)

    def test_invalid_state_raises(self):
        with self.assertRaises(AttributeError):
            post_localSGD_hook(object(), object())


if __name__ == "__main__":
    run_tests()
