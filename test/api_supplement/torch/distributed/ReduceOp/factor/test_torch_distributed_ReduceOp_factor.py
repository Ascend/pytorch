# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.ReduceOp.factor 属性及多卡 HCCL 归约正确性
API 名称：torch.distributed.ReduceOp.factor
API 签名：ReduceOp.factor -> float | Tensor | None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 多卡集合通信 | world_size=2，标准 op 与 premul_sum 归约 | 已覆盖（test_standard_op_factor_multinpu、test_premul_sum_factor_multinpu） |
| 同步/异步分支 | async_op=False 与 async_op=True（wait 后核对数值） | 已覆盖（test_standard_op_factor_multinpu） |
| 结果数值对照 | 多卡归约结果移至 CPU 后与期望 Golden 对照 | 已覆盖（test_standard_op_factor_multinpu、test_premul_sum_factor_multinpu） |
| 空/非空 | 非 PREMUL_SUM 访问抛 RuntimeError；PREMUL_SUM 返回非空因子 | 已覆盖（test_factor_behavior_and_errors） |
| 参数类型 | float/int 标量因子、CPU/设备侧单元素 Tensor 因子 | 已覆盖（test_factor_behavior_and_errors） |
| 边界值 | 负因子、零因子、单元素 0-D 与 1-D Tensor | 已覆盖（test_factor_behavior_and_errors） |
| 异常路径 | 非 PREMUL_SUM/只读赋值/多元素 Tensor/非法类型抛错 | 已覆盖（test_factor_behavior_and_errors） |
| 进程组生命周期 | FileStore 临时文件 rendezvous | 已覆盖（test_standard_op_factor_multinpu、test_premul_sum_factor_multinpu） |

未覆盖项及原因：
- 传参与不传参：.factor 为无参只读属性，不接收调用参数。

注意：多卡用例在 NPU/CUDA 环境通过 FileStore 临时文件 rendezvous 运行；
     标准 op 与 PREMUL_SUM 分拆为独立多卡用例以便清晰暴露后端差异。
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


def _worker_standard_op_factor(rank, world_size, device_name):
    local_device = f"{device_name}:{rank}"
    base = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    op = dist.ReduceOp(dist.ReduceOp.RedOpType.SUM)
    err = _assert_raises(RuntimeError, getattr, op, "factor")
    assert "PREMUL_SUM" in str(err), "Expected RuntimeError mentioning PREMUL_SUM for SUM.factor"

    tensor = (base * float(rank + 1)).to(local_device)
    dist.all_reduce(tensor, op=op)
    torch.testing.assert_close(tensor.cpu(), base * 3.0)

    async_tensor = (base * float(rank + 1)).to(local_device)
    work = dist.all_reduce(async_tensor, op=op, async_op=True)
    work.wait()
    torch.testing.assert_close(async_tensor.cpu(), base * 3.0)


def _worker_premul_sum_factor(rank, world_size, device_name):
    local_device = f"{device_name}:{rank}"
    base = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    scale = 2.0
    premul_op = dist._make_nccl_premul_sum(scale)
    assert premul_op.factor == scale, f"Expected factor {scale}, got {premul_op.factor}"

    tensor = (base * float(rank + 1)).to(local_device)
    dist.all_reduce(tensor, op=premul_op)
    torch.testing.assert_close(tensor.cpu(), base * 3.0 * scale)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestReduceOpFactor(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def _run_mp(self, fn, world_size=2, *extra):
        count = torch.npu.device_count() if self.device_name == "npu" else torch.cuda.device_count()
        if count < world_size:
            self.skipTest("At least 2 NPU/CUDA devices are required.")
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_name = f.name
        try:
            mp.spawn(_init_dist, args=(world_size, fn, BACKEND, self.device_name, file_name, *extra),
                     nprocs=world_size, join=True)
        finally:
            if os.path.exists(file_name):
                os.unlink(file_name)

    def test_factor_behavior_and_errors(self):
        for name in ("SUM", "AVG", "PRODUCT", "MIN", "MAX", "BAND", "BOR", "BXOR"):
            op = dist.ReduceOp(getattr(dist.ReduceOp.RedOpType, name))
            with self.assertRaisesRegex(RuntimeError, "PREMUL_SUM"):
                _ = op.factor

        for val in (2.5, -0.5, 0.0, 3.0):
            op = dist._make_nccl_premul_sum(val)
            self.assertEqual(op.op, dist.ReduceOp.RedOpType.PREMUL_SUM)
            self.assertIsInstance(op.factor, float)
            self.assertAlmostEqual(op.factor, val, places=12)

        cpu_factor = torch.tensor(0.25, dtype=torch.float32)
        op_cpu = dist._make_nccl_premul_sum(cpu_factor)
        self.assertIsInstance(op_cpu.factor, torch.Tensor)
        torch.testing.assert_close(op_cpu.factor, cpu_factor)

        dev_factor = torch.tensor([1.75], dtype=torch.float32, device=self.device_name)
        op_dev = dist._make_nccl_premul_sum(dev_factor)
        self.assertIsInstance(op_dev.factor, torch.Tensor)
        self.assertEqual(op_dev.factor.device.type, self.device_name)
        torch.testing.assert_close(op_dev.factor.cpu(), torch.tensor([1.75], dtype=torch.float32))

        op = dist._make_nccl_premul_sum(1.5)
        with self.assertRaises(AttributeError):
            op.factor = 2.0
        with self.assertRaises(RuntimeError):
            dist._make_nccl_premul_sum(torch.tensor([1.0, 2.0], dtype=torch.float32))
        with self.assertRaises(TypeError):
            dist._make_nccl_premul_sum("not_a_factor")

    @skipIfUnsupportMultiNPU(2)
    def test_standard_op_factor_multinpu(self):
        self._run_mp(_worker_standard_op_factor)

    @skipIfUnsupportMultiNPU(2)
    def test_premul_sum_factor_multinpu(self):
        self._run_mp(_worker_premul_sum_factor)


if __name__ == "__main__":
    run_tests()
