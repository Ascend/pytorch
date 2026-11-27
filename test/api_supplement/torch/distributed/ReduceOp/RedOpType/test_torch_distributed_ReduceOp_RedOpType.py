# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.ReduceOp.RedOpType 枚举及多卡 HCCL 归约正确性
API 名称：torch.distributed.ReduceOp.RedOpType
API 签名：class torch.distributed.ReduceOp.RedOpType

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 多卡集合通信 | world_size=2，HCCL/NCCL 后端真实 all_reduce | 已覆盖（test_all_reduce_redoptype_multinpu） |
| 同步/异步分支 | async_op=False 与 async_op=True（wait 后核对数值） | 已覆盖（test_all_reduce_redoptype_multinpu） |
| 结果数值对照 | SUM/PRODUCT/MIN/MAX/AVG 各 rank 结果与期望值一致 | 已覆盖（test_all_reduce_redoptype_multinpu） |
| 枚举选项 | SUM/AVG/PRODUCT/MIN/MAX/BAND/BOR/BXOR/PREMUL_SUM | 已覆盖（test_enum_members_and_roundtrip） |
| 参数类型 | 整数构造、与 ReduceOp 互转及非法入参报错 | 已覆盖（test_enum_members_and_roundtrip、test_invalid_enum_value） |
| 进程组生命周期 | FileStore 临时文件 rendezvous，init/destroy 正常 | 已覆盖（test_all_reduce_redoptype_multinpu） |
| 异常路径 | 非法枚举值/非法类型构造抛 ValueError/TypeError | 已覆盖（test_invalid_enum_value） |

未覆盖项及原因：
- 位运算（BAND/BOR/BXOR）多卡 HCCL 通信：浮点张量不支持位归约，已在枚举成员完整性用例中覆盖其定义。

注意：多卡用例在 NPU/CUDA 环境通过 FileStore 临时文件 rendezvous 运行，
     通信结果移至 CPU 后以 torch.testing.assert_close 对照各 rank 期望 Golden。
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


def _worker_all_reduce_redoptypes(rank, world_size, device_name):
    local_device = f"{device_name}:{rank}"
    base = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    cases = (
        (dist.ReduceOp.RedOpType.SUM, base * 3.0),
        (dist.ReduceOp.RedOpType.PRODUCT, base * base * 2.0),
        (dist.ReduceOp.RedOpType.MIN, base * 1.0),
        (dist.ReduceOp.RedOpType.MAX, base * 2.0),
        (dist.ReduceOp.RedOpType.AVG, base * 1.5),
    )
    for op_type, expected in cases:
        tensor = (base * float(rank + 1)).to(local_device)
        dist.all_reduce(tensor, op=op_type)
        torch.testing.assert_close(tensor.cpu(), expected)

    async_tensor = (base * float(rank + 1)).to(local_device)
    work = dist.all_reduce(async_tensor, op=dist.ReduceOp(dist.ReduceOp.RedOpType.SUM), async_op=True)
    work.wait()
    torch.testing.assert_close(async_tensor.cpu(), base * 3.0)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestReduceOpRedOpType(TestCase):

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

    def test_enum_members_and_roundtrip(self):
        red_op_type = dist.ReduceOp.RedOpType
        members = red_op_type.__members__
        for name in ("SUM", "AVG", "PRODUCT", "MIN", "MAX", "BAND", "BOR", "BXOR", "PREMUL_SUM"):
            self.assertIn(name, members)
            member = getattr(red_op_type, name)
            self.assertIsInstance(member, red_op_type)
            self.assertEqual(member, members[name])
            self.assertEqual(getattr(dist.ReduceOp, name), member)
            self.assertEqual(red_op_type(int(member)), member)
            # PREMUL_SUM requires _make_nccl_premul_sum instead of direct ReduceOp(member).
            if name != "PREMUL_SUM":
                wrapped = dist.ReduceOp(member)
                self.assertIsInstance(wrapped, dist.ReduceOp)
                self.assertEqual(wrapped.op, member)
            else:
                with self.assertRaises(RuntimeError):
                    dist.ReduceOp(member)
                premul = dist._make_nccl_premul_sum(2.0)
                self.assertIsInstance(premul, dist.ReduceOp)
                self.assertEqual(premul.op, member)

        self.assertNotEqual(red_op_type.SUM, red_op_type.MAX)
        self.assertEqual(red_op_type.SUM, dist.ReduceOp.SUM)

    def test_invalid_enum_value(self):
        with self.assertRaises((ValueError, TypeError)):
            dist.ReduceOp.RedOpType(99999)
        with self.assertRaises(TypeError):
            dist.ReduceOp.RedOpType("SUM")

    @skipIfUnsupportMultiNPU(2)
    def test_all_reduce_redoptype_multinpu(self):
        self._run_mp(_worker_all_reduce_redoptypes)


if __name__ == "__main__":
    run_tests()
