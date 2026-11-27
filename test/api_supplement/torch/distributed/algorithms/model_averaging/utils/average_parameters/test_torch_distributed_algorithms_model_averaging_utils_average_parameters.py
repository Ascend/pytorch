# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.algorithms.model_averaging.utils.average_parameters 接口功能正确性
API 名称：torch.distributed.algorithms.model_averaging.utils.average_parameters
API 签名：average_parameters(params: Iterator[Parameter], process_group: ProcessGroup)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|-------------------------------------|--------------------------------------------------|
| 多卡集合通信   | world_size=2，HCCL 后端真实参数平均 | 已覆盖（test_explicit_pg_and_rank_not_in_group、test_mixed_dtypes_preserved） |
| 传参与不传参   | 显式子组、单卡子组与 WORLD 进程组 | 已覆盖（test_explicit_pg_and_rank_not_in_group、test_mixed_dtypes_preserved） |
| 结果数值对照   | 平均后各 rank 参数精确等于两卡均值且形状不变 | 已覆盖（test_explicit_pg_and_rank_not_in_group、test_mixed_dtypes_preserved） |
| 参数类型/边界  | 混合 dtype（fp32/fp16）与多维张量，恢复原类型 | 已覆盖（test_mixed_dtypes_preserved） |
| 进程组生命周期 | FileStore 临时文件 init/destroy_process_group | 已覆盖（全部多卡用例） |
| 异常路径       | 空迭代器与混合设备（设备/CPU）触发 RuntimeError | 已覆盖（test_invalid_params_raises） |

未覆盖项及原因：
- process_group=None 多层 nn.Sequential 展平平均：NPU A5 上 HcclAllreduce 触发 AICPU 507018 异常，暂未启用。

注意：多卡用例在 2 卡 NPU/CUDA 环境通过 HCCL/NCCL 运行，以各 rank 初始参数的解析均值为 Golden；
     同时覆盖非组成员 rank 跳过更新、混合 dtype 保持原类型及空参数/异构设备异常路径。
"""
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.algorithms.model_averaging.utils import average_parameters

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


def _worker_explicit_pg_and_rank_not_in_group(rank, world_size, device_name):
    local_device = f"{device_name}:{rank}"
    sub_pg = dist.new_group(ranks=list(range(world_size)))
    p1 = nn.Parameter(torch.full((3, 5), float(rank + 1), device=local_device))
    p2 = nn.Parameter(torch.full((2, 3, 4), float(10 * (rank + 1)), device=local_device))
    average_parameters(iter([p1, p2]), process_group=sub_pg)
    torch.testing.assert_close(p1.data, torch.full((3, 5), 1.5, device=local_device))
    torch.testing.assert_close(p2.data, torch.full((2, 3, 4), 15.0, device=local_device))

    rank0_only_pg = dist.new_group(ranks=[0])
    init_val = float(10 + 5 * rank)
    p3 = nn.Parameter(torch.full((4, 4), init_val, device=local_device))
    average_parameters(iter([p3]), process_group=rank0_only_pg)
    torch.testing.assert_close(p3.data, torch.full((4, 4), init_val, device=local_device))


def _worker_mixed_dtypes_preserved(rank, world_size, device_name):
    local_device = f"{device_name}:{rank}"
    p_fp32 = nn.Parameter(torch.full((4, 4), float(2 + 4 * rank), device=local_device, dtype=torch.float32))
    p_fp16 = nn.Parameter(torch.full((6,), float(1 + 3 * rank), device=local_device, dtype=torch.float16))
    average_parameters(iter([p_fp32, p_fp16]), process_group=dist.group.WORLD)
    assert p_fp32.dtype == torch.float32, f"Expected float32, got {p_fp32.dtype}"
    assert p_fp16.dtype == torch.float16, f"Expected float16, got {p_fp16.dtype}"
    torch.testing.assert_close(p_fp32.data.cpu(), torch.full((4, 4), 4.0, dtype=torch.float32))
    torch.testing.assert_close(p_fp16.data.cpu(), torch.full((6,), 2.5, dtype=torch.float16), rtol=1e-2, atol=1e-2)


def _worker_invalid_params_raises(rank, world_size, device_name):
    _assert_raises((RuntimeError, ValueError), average_parameters, iter([]), process_group=dist.group.WORLD)
    p_dev = nn.Parameter(torch.ones(4, device=f"{device_name}:{rank}"))
    p_cpu = nn.Parameter(torch.ones(4, device="cpu"))
    _assert_raises(RuntimeError, average_parameters, iter([p_dev, p_cpu]), process_group=dist.group.WORLD)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestAverageParameters(TestCase):
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
    def test_explicit_pg_and_rank_not_in_group(self):
        self._run_mp(_worker_explicit_pg_and_rank_not_in_group)

    @skipIfUnsupportMultiNPU(2)
    def test_mixed_dtypes_preserved(self):
        self._run_mp(_worker_mixed_dtypes_preserved)

    @skipIfUnsupportMultiNPU(2)
    def test_invalid_params_raises(self):
        self._run_mp(_worker_invalid_params_raises)


if __name__ == "__main__":
    run_tests()
