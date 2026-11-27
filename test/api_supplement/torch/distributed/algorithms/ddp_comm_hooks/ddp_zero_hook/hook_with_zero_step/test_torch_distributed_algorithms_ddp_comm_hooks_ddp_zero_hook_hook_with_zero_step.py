# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook.hook_with_zero_step 接口功能正确性
API 名称：torch.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook.hook_with_zero_step
API 签名：hook_with_zero_step(hook, ddp, zero, shard_buckets=False)

覆盖维度表：
| 覆盖维度       | 说明                                         | 覆盖情况                                              |
|----------------|--------------------------------------------|-----------------------------------------------------|
| 多卡集合通信   | world_size=2，HCCL 后端 DDP+ZeRO 重叠通信    | 已覆盖（test_shard_buckets_false）                    |
| 参数组合       | shard_buckets=False（默认）                  | 已覆盖（test_shard_buckets_false）                    |
| 结果数值对照   | 重叠优化器更新后的参数与标准 DDP+SGD 对齐    | 已覆盖（test_shard_buckets_false）                    |
| 跨 rank 一致性 | 各 rank 广播后模型参数完全一致               | 已覆盖（test_shard_buckets_false）                    |
| 进程组生命周期 | init / destroy_process_group 正常            | 已覆盖（全部多卡用例）                                |
| rendezvous     | FileStore 临时文件（无 MASTER_PORT）         | 已覆盖（全部多卡用例）                                |
| 异常路径       | overlap_with_ddp=False 触发 ValueError       | 已覆盖（test_overlap_false_raises_value_error）       |

未覆盖项及原因：
- Gloo 后端 RuntimeError 路径：本套用例统一使用 HCCL/NCCL 多卡后端，未额外构造 Gloo 进程组。
- shard_buckets=True：精简重复多卡用例，聚焦默认路径与非法配置异常。

注意：多卡用例在 NPU/GPU 环境运行，重叠优化器在 static_graph=False 下前两次迭代用于
     重建与收集 bucket 信息，第 3、4 次迭代执行参数更新；结果与标准 DDP+SGD 对照。
"""
import copy
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook import hook_with_zero_step
from torch.distributed.optim import ZeroRedundancyOptimizer
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


def _build_model(device):
    torch.manual_seed(1234)
    return nn.Sequential(nn.Linear(8, 8, bias=False), nn.ReLU(), nn.Linear(8, 4, bias=False)).to(device)


def _worker_overlap_check(rank, world_size, device_name):
    dev = torch.device(f"{device_name}:{rank}")
    base_model = _build_model(dev)
    init_weights = [p.detach().clone() for p in base_model.parameters()]
    ref_base = copy.deepcopy(base_model)

    # Small bucket_cap_mb forces multiple buckets across layers.
    ddp_model = DDP(base_model, device_ids=[rank], bucket_cap_mb=0.0001)
    zero = ZeroRedundancyOptimizer(
        ddp_model.parameters(), optimizer_class=torch.optim.SGD, overlap_with_ddp=True, lr=0.1
    )
    comm_hook = hook_with_zero_step(default_hooks.allreduce_hook, ddp_model, zero, shard_buckets=False)
    ddp_model.register_comm_hook(state=None, hook=comm_hook)

    ref_ddp = DDP(ref_base, device_ids=[rank])
    ref_opt = torch.optim.SGD(ref_ddp.parameters(), lr=0.1)
    x = torch.ones(2, 8, device=dev) * float(rank + 1)

    # First 2 iterations rebuild/save DDP buckets; iterations 2 and 3 update params.
    for step_idx in range(4):
        zero.zero_grad()
        ddp_model(x).sum().backward()
        if step_idx < 2:
            for p, p0 in zip(ddp_model.parameters(), init_weights):
                torch.testing.assert_close(p.detach().cpu(), p0.cpu())

    for _ in range(2):
        ref_opt.zero_grad()
        ref_ddp(x).sum().backward()
        ref_opt.step()

    for p, p0, p_ref in zip(ddp_model.parameters(), init_weights, ref_ddp.parameters()):
        assert not torch.allclose(p.detach().cpu(), p0.cpu()), "Parameters were not updated"
        p_r0 = p.detach().clone()
        dist.broadcast(p_r0, src=0)
        torch.testing.assert_close(p.detach().cpu(), p_r0.cpu())
        torch.testing.assert_close(p.detach().cpu(), p_ref.detach().cpu(), rtol=1e-4, atol=1e-4)


def _worker_overlap_false_raises(rank, world_size, device_name):
    dev = torch.device(f"{device_name}:{rank}")
    ddp_model = DDP(_build_model(dev), device_ids=[rank])
    zero = ZeroRedundancyOptimizer(
        ddp_model.parameters(), optimizer_class=torch.optim.SGD, overlap_with_ddp=False, lr=0.1
    )
    err = _assert_raises(ValueError, hook_with_zero_step, default_hooks.allreduce_hook, ddp_model, zero)
    assert "overlap_with_ddp=True" in str(err)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestHookWithZeroStep(TestCase):

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
    def test_shard_buckets_false(self):
        self._run_mp(_worker_overlap_check)

    @skipIfUnsupportMultiNPU(2)
    def test_overlap_false_raises_value_error(self):
        self._run_mp(_worker_overlap_false_raises)


if __name__ == "__main__":
    run_tests()
