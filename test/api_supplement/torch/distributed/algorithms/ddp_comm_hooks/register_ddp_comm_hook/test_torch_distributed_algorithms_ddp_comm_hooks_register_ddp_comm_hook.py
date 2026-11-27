# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.algorithms.ddp_comm_hooks.register_ddp_comm_hook 接口功能正确性
API 名称：torch.distributed.algorithms.ddp_comm_hooks.register_ddp_comm_hook
API 签名：register_ddp_comm_hook(comm_hook_type: DDPCommHookType, model, state=None)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|-------------------------------------|--------------------------------------------------|
| 多卡集合通信   | world_size=2，HCCL 后端真实 DDP 注册反传 | 已覆盖（test_allreduce_default_and_explicit_state 等） |
| 枚举选项       | DDPCommHookType 主要成员 | 已覆盖（test_compress_quantize_powersgd_and_noop_hooks） |
| 传参与不传参   | state 省略默认（None）与显式传入进程组 | 已覆盖（test_allreduce_default_and_explicit_state） |
| 结果数值对照   | 聚合钩子梯度等于两卡均值；NOOP 保留本地梯度 | 已覆盖（全部正常多卡用例） |
| 进程组生命周期 | FileStore 临时文件 init/destroy_process_group | 已覆盖（全部多卡用例） |
| 异常路径       | 非法 comm_hook_type 与非 DDP model 抛异常 | 已覆盖（test_invalid_args） |

未覆盖项及原因：
- BF16_COMPRESS：NPU 上 DDP._check_comm_hook() 尚未放行 NPU/HCCL 注册 BF16_COMPRESS，暂未在多卡用例中启用。

注意：多卡用例在 2 卡 NPU/CUDA 环境通过 register_ddp_comm_hook 注册不同 DDPCommHookType，
     并断言反传梯度与理论均值（NOOP 断言保留本地未聚合梯度）一致。
"""
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.algorithms.ddp_comm_hooks import DDPCommHookType, register_ddp_comm_hook
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

_OMIT = object()


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


def _run_ddp_step(rank, device_name, hook_type, state=_OMIT, seed=0, in_dim=8, out_dim=4, batch=2, bias=False):
    dev = f"{device_name}:{rank}"
    torch.manual_seed(seed)
    model = nn.Linear(in_dim, out_dim, bias=bias).to(dev)
    ddp = DDP(model, device_ids=[rank])
    if state is _OMIT:
        register_ddp_comm_hook(hook_type, ddp)
    else:
        register_ddp_comm_hook(hook_type, ddp, state=state)
    ddp(torch.full((batch, in_dim), float(1 + 2 * rank), device=dev)).sum().backward()
    return model, dev


def _worker_allreduce_default_and_explicit_state(rank, world_size, device_name):
    sub_pg = dist.new_group(ranks=list(range(world_size)))
    for state in (_OMIT, sub_pg):
        model, dev = _run_ddp_step(rank, device_name, DDPCommHookType.ALLREDUCE, state=state, seed=0, bias=True)
        torch.testing.assert_close(model.weight.grad, torch.full((4, 8), 4.0, device=dev))
        torch.testing.assert_close(model.bias.grad, torch.full((4,), 2.0, device=dev))


def _worker_compress_quantize_powersgd_and_noop_hooks(rank, world_size, device_name):
    model, dev = _run_ddp_step(rank, device_name, DDPCommHookType.FP16_COMPRESS, state=dist.group.WORLD, seed=1)
    torch.testing.assert_close(model.weight.grad, torch.full((4, 8), 4.0, device=dev), rtol=1e-2, atol=1e-2)

    # NOOP skips averaging: rank 0 local grad=2.0, rank 1 local grad=6.0.
    model, dev = _run_ddp_step(rank, device_name, DDPCommHookType.NOOP, seed=2)
    torch.testing.assert_close(model.weight.grad, torch.full((4, 8), float(2 * (1 + 2 * rank)), device=dev))

    model, dev = _run_ddp_step(
        rank, device_name, DDPCommHookType.QUANTIZE_PER_TENSOR, state=dist.group.WORLD, seed=3,
        in_dim=16, out_dim=8, batch=4,
    )
    torch.testing.assert_close(model.weight.grad, torch.full((8, 16), 8.0, device=dev), rtol=0.05, atol=0.15)

    model, dev = _run_ddp_step(rank, device_name, DDPCommHookType.POWER_SGD, state=dist.group.WORLD, seed=4)
    torch.testing.assert_close(model.weight.grad, torch.full((4, 8), 4.0, device=dev))


def _worker_invalid_args(rank, world_size, device_name):
    model = nn.Linear(4, 2).to(f"{device_name}:{rank}")
    ddp = DDP(model, device_ids=[rank])
    _assert_raises(AttributeError, register_ddp_comm_hook, "ALLREDUCE", ddp)
    _assert_raises(AttributeError, register_ddp_comm_hook, DDPCommHookType.ALLREDUCE, model)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRegisterDdpCommHook(TestCase):

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
    def test_allreduce_default_and_explicit_state(self):
        self._run_mp(_worker_allreduce_default_and_explicit_state)

    @skipIfUnsupportMultiNPU(2)
    def test_compress_quantize_powersgd_and_noop_hooks(self):
        self._run_mp(_worker_compress_quantize_powersgd_and_noop_hooks)

    @skipIfUnsupportMultiNPU(2)
    def test_invalid_args(self):
        self._run_mp(_worker_invalid_args)


if __name__ == "__main__":
    run_tests()
