# -*- coding: utf-8 -*-
"""
测试目的：验证 fp16_compress_wrapper 接口功能正确性
API 名称：torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper
API 签名：fp16_compress_wrapper(hook)

覆盖维度表：
| 覆盖维度       | 说明                                         | 覆盖情况                                              |
|----------------|--------------------------------------------|-----------------------------------------------------|
| 多卡集合通信   | world_size=2，HCCL 后端包装通信钩子          | 已覆盖（test_custom_hook_sees_fp16）                  |
| 压缩与解压语义 | 内层钩子接收 float16，最终梯度恢复 float32   | 已覆盖（test_custom_hook_sees_fp16）                  |
| 结果数值对照   | 聚合后梯度与理论均值一致                     | 已覆盖（test_custom_hook_sees_fp16）                  |
| 进程组生命周期 | init / destroy_process_group 正常            | 已覆盖（全部多卡用例）                                |
| rendezvous     | FileStore 临时文件（无 MASTER_PORT）         | 已覆盖（全部多卡用例）                                |
| 异常路径       | 传入非可调用 hook 在执行时触发 TypeError     | 已覆盖（test_non_callable_hook_raises）               |

未覆盖项及原因：
- 自定义 new_group 与等价性对照单测：精简冗余多卡用例，聚焦核心压缩解压验证与非法 hook 异常。

注意：多卡用例在 NPU/GPU 环境运行，验证包装器向内层钩子传递 float16 buffer 并在
     回调后恢复 float32 梯度，且数值与理论均值一致。
"""
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
    allreduce_hook,
    fp16_compress_wrapper,
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


def _assert_raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as err:
        return err
    raise AssertionError(f"Expected {exc.__name__}")


def _worker_custom_hook_sees_fp16(rank, world_size, device_name):
    seen_dtypes = []

    def recording_hook(state, bucket):
        seen_dtypes.append(bucket.buffer().dtype)
        return allreduce_hook(state, bucket)

    dev = torch.device(f"{device_name}:{rank}")
    ddp = DDP(nn.Linear(4, 2, bias=False).to(dev), device_ids=[rank])
    wrapped = fp16_compress_wrapper(recording_hook)
    assert callable(wrapped)
    ddp.register_comm_hook(state=None, hook=wrapped)
    ddp(torch.full((1, 4), 2.0 if rank == 0 else 4.0, device=dev)).sum().backward()
    grad = ddp.module.weight.grad
    assert seen_dtypes and all(dt == torch.float16 for dt in seen_dtypes), f"Got {seen_dtypes}"
    assert grad.dtype == torch.float32
    torch.testing.assert_close(grad.cpu(), torch.full((2, 4), 3.0), rtol=1e-3, atol=1e-3)


def _worker_non_callable_hook_raises(rank, world_size, device_name):
    dev = torch.device(f"{device_name}:{rank}")
    ddp = DDP(nn.Linear(4, 2, bias=False).to(dev), device_ids=[rank])
    ddp.register_comm_hook(state=None, hook=fp16_compress_wrapper("not_callable"))
    x = torch.ones(1, 4, device=dev)
    _assert_raises(TypeError, lambda: ddp(x).sum().backward())


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestFp16CompressWrapper(TestCase):

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
    def test_custom_hook_sees_fp16(self):
        self._run_mp(_worker_custom_hook_sees_fp16)

    @skipIfUnsupportMultiNPU(2)
    def test_non_callable_hook_raises(self):
        self._run_mp(_worker_non_callable_hook_raises)


if __name__ == "__main__":
    run_tests()
