# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.ReduceOp.RedOpType.name 属性及多卡 HCCL 归约正确性
API 名称：torch.distributed.ReduceOp.RedOpType.name
API 签名：ReduceOp.RedOpType.name -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 多卡集合通信 | world_size=2，按 op.name 分发 HCCL/NCCL all_reduce | 已覆盖（test_dispatch_by_name_multinpu） |
| 同步/异步分支 | async_op=False 与 async_op=True（wait 后核对数值） | 已覆盖（test_dispatch_by_name_multinpu） |
| 结果数值对照 | 按 op.name 计算期望值并与归约结果对照 | 已覆盖（test_dispatch_by_name_multinpu） |
| 枚举选项 | 全部 RedOpType 成员的 .name 返回值 | 已覆盖（test_all_member_names_match_keys） |
| 参数类型 | 返回类型为 str 且可经 getattr 反查原成员 | 已覆盖（test_all_member_names_match_keys） |
| 进程组生命周期 | FileStore 临时文件 rendezvous，init/destroy 正常 | 已覆盖（test_dispatch_by_name_multinpu） |
| 异常路径 | 只读属性赋值抛 AttributeError | 已覆盖（test_name_is_readonly） |

未覆盖项及原因：
- 空/非空入参：.name 为无参只读属性，不接收输入参数。

注意：多卡用例在 NPU/CUDA 环境通过 FileStore 临时文件 rendezvous 运行，
     通信结果移至 CPU 后以 torch.testing.assert_close 对照期望 Golden。
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


def _worker_dispatch_by_name(rank, world_size, device_name):
    local_device = f"{device_name}:{rank}"
    base = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float32)
    expected_by_name = {"SUM": base * 3.0, "MIN": base * 1.0, "MAX": base * 2.0}

    for op in (dist.ReduceOp.RedOpType.SUM, dist.ReduceOp.RedOpType.MIN, dist.ReduceOp.RedOpType.MAX):
        op_name = op.name
        assert op_name in expected_by_name, f"Unexpected op name: {op_name}"
        resolved = getattr(dist.ReduceOp.RedOpType, op_name)
        tensor = (base * float(rank + 1)).to(local_device)
        dist.all_reduce(tensor, op=resolved)
        torch.testing.assert_close(tensor.cpu(), expected_by_name[op_name])

    sum_op = getattr(dist.ReduceOp.RedOpType, dist.ReduceOp.RedOpType.SUM.name)
    async_tensor = (base * float(rank + 1)).to(local_device)
    work = dist.all_reduce(async_tensor, op=sum_op, async_op=True)
    work.wait()
    torch.testing.assert_close(async_tensor.cpu(), expected_by_name["SUM"])


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestReduceOpRedOpTypeName(TestCase):

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

    def test_all_member_names_match_keys(self):
        red_op_type = dist.ReduceOp.RedOpType
        members = red_op_type.__members__
        self.assertGreater(len(members), 0)
        seen_names = set()
        for key, member in members.items():
            self.assertIsInstance(member.name, str)
            self.assertEqual(member.name, key)
            self.assertEqual(getattr(red_op_type, member.name), member)
            seen_names.add(member.name)

        for expected in ("SUM", "AVG", "PRODUCT", "MIN", "MAX", "BAND", "BOR", "BXOR", "PREMUL_SUM"):
            self.assertIn(expected, seen_names)
            self.assertEqual(getattr(red_op_type, expected).name, expected)

    def test_name_is_readonly(self):
        with self.assertRaises(AttributeError):
            dist.ReduceOp.RedOpType.SUM.name = "CUSTOM"

    @skipIfUnsupportMultiNPU(2)
    def test_dispatch_by_name_multinpu(self):
        self._run_mp(_worker_dispatch_by_name)


if __name__ == "__main__":
    run_tests()
