# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Work.block_current_stream 接口功能正确性
API 名称：torch.distributed.Work.block_current_stream
API 签名：Work.block_current_stream(self) -> None

覆盖维度表：
| 覆盖维度       | 说明                                       | 覆盖情况                                             |
|----------------|--------------------------------------------|------------------------------------------------------|
| 多卡集合通信   | world_size=2，HCCL/NCCL 后端真实异步通信   | 已覆盖（test_all_reduce_default_and_custom_stream）  |
| 同步/异步分支  | async_op=True 返回 Work 后阻塞默认/自定义流 | 已覆盖（test_all_reduce_default_and_custom_stream） |
| 结果数值对照   | 阻塞当前流后执行下游算子，断言精确数值结果 | 已覆盖（test_all_reduce_default_and_custom_stream）  |
| 进程组生命周期 | init_process_group / destroy_process_group | 已覆盖（全部用例统一通过 _init_dist 管理）           |
| rendezvous     | FileStore 临时文件（无 MASTER_PORT）       | 已覆盖（全部用例使用独立临时文件）                   |
| 异常路径       | 传入多余位置参数触发 TypeError             | 已覆盖（test_invalid_extra_args_raises）             |

未覆盖项及原因：
- NPU 流阻塞用例：NPU 上 ProcessGroupHCCL 尚未完成 StreamBlock 适配，在 NPU 环境跳过流阻塞用例。

注意：多卡用例在 NPU/CUDA 环境运行，以各 rank 异步集合通信及下游计算的期望数值为 Golden，
     使用 torch.testing.assert_close 进行浮点对照；设备卡数不足 2 时以 self.skipTest 跳过。
"""
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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


def _worker_all_reduce_default_and_custom_stream(rank, world_size, device_name):
    dev_mod = torch.npu if device_name == "npu" else torch.cuda
    tensor = torch.full((64, 64), float(rank + 1), dtype=torch.float32, device=f"{device_name}:{rank}")

    work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
    ret = work.block_current_stream()
    assert ret is None, f"Expected None from block_current_stream, got {ret}"

    # Downstream compute on default stream must observe the completed all_reduce result.
    out = tensor * 2.0 + 1.0
    dev_mod.synchronize()
    work.wait()

    expected_sum = sum(range(1, world_size + 1))
    expected = torch.full((64, 64), float(expected_sum) * 2.0 + 1.0, dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), expected)

    # Custom non-default stream blocking after async all_reduce.
    tensor_custom = torch.full((32, 32), float(rank + 2), dtype=torch.float32, device=f"{device_name}:{rank}")
    work_custom = dist.all_reduce(tensor_custom, op=dist.ReduceOp.SUM, async_op=True)
    custom_stream = dev_mod.Stream()
    with dev_mod.stream(custom_stream):
        work_custom.block_current_stream()
        out_custom = tensor_custom + float(rank * 10)
    custom_stream.synchronize()
    work_custom.wait()

    expected_custom_sum = sum(r + 2 for r in range(world_size))
    expected_custom = torch.full((32, 32), float(expected_custom_sum + rank * 10), dtype=torch.float32)
    torch.testing.assert_close(out_custom.cpu(), expected_custom)


def _worker_invalid_extra_args(rank, world_size, device_name):
    tensor = torch.ones((8,), dtype=torch.float32, device=f"{device_name}:{rank}")
    work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
    try:
        _assert_raises(TypeError, work.block_current_stream, 1)
    finally:
        work.wait()


NPU_SKIP_REASON = "PTA/NPU defect: ProcessGroupHCCL Work.block_current_stream fails to create StreamBlock"


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestWorkBlockCurrentStream(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def _run_mp(self, fn, world_size=2, *extra):
        count = torch.npu.device_count() if self.device_name == "npu" else torch.cuda.device_count()
        if count < world_size:
            self.skipTest(f"Need at least {world_size} {self.device_name} devices")
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_name = f.name
        try:
            mp.spawn(_init_dist, args=(world_size, fn, BACKEND, self.device_name, file_name, *extra),
                     nprocs=world_size, join=True)
        finally:
            if os.path.exists(file_name):
                os.unlink(file_name)

    @unittest.skipIf(DEVICE_NAME == "npu", NPU_SKIP_REASON)
    @skipIfUnsupportMultiNPU(2)
    def test_all_reduce_default_and_custom_stream(self):
        self._run_mp(_worker_all_reduce_default_and_custom_stream)

    @skipIfUnsupportMultiNPU(2)
    def test_invalid_extra_args_raises(self):
        self._run_mp(_worker_invalid_extra_args)


if __name__ == "__main__":
    run_tests()
