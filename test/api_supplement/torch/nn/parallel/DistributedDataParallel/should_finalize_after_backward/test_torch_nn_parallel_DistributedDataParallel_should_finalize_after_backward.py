# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.parallel.DistributedDataParallel.should_finalize_after_backward 接口功能正确性
API 名称：torch.nn.parallel.DistributedDataParallel.should_finalize_after_backward
API 签名：should_finalize_after_backward(self) -> bool（property）
         （手动 finalization 下指示 finalize_backward 的调用时机：True 时须在 backward
          返回后调用，False 时可在桶就绪 hook 中调用）

覆盖维度表：
| 覆盖维度         | 说明                                                  | 覆盖情况                                        |
|------------------|-------------------------------------------------------|-------------------------------------------------|
| 多卡集合通信     | world_size=2，HCCL 后端真实梯度 all-reduce            | 已覆盖（三个多卡用例）                          |
| 参数类型         | property 返回 bool                                    | 已覆盖（false_when_disabled / rebuild）         |
| 传参与不传参     | 只读属性，随手动模式开关与桶重建阶段变化              | 已覆盖（全部用例）                              |
| 等价类/边界值    | 关闭手动模式（恒 False）与桶重建两阶段（True→False）  | 已覆盖（false_when_disabled / rebuild）         |
| 正常传参场景     | 手动模式首迭代（桶重建期）为 True；重建完成后为 False | 已覆盖（rebuild_true / rebuild_completed）      |
| 异常传参场景     | 不适用：只读查询属性，无异常路径                      | 不适用                                          |
| 混合设备类型     | 不适用：无 Tensor 入参                                | 不适用                                          |

未覆盖项及原因：
- native mixed precision（DDP 混合精度）分支：未覆盖。契约中
  mixed_precision is not None 恒为 True 的分支需搭配 DDP 混合精度配置，
  本批未构造该配置，不凭空断言。
- 内存：不适用。查询属性无分配语义。
- 反向：不适用。查询属性本身无梯度语义（其驱动的 finalize 写回由
  finalize_backward 用例看护）。

注意：多卡用例在 NPU 环境运行（HCCL）；取值时序为实测固定契约
     （torch_npu 2.15.0.dev20261118 + 2.15.0 源码）：关闭手动模式恒为
     False；开启后首迭代（含 forward 前至 finalize_backward 完成）处于
     桶重建期返回 True，第二次 backward 起返回 False。device_count < 2
     时以 skipTest 归入环境不具备，不伪造对照。
"""
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Resolve device only when tests actually run; keep default CI collection cheap.
DEVICE_NAME = 'cpu'
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = 'npu'
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

    def skipIfUnsupportMultiNPU(n):
        def deco(fn):
            return fn
        return deco


BACKEND = 'hccl' if DEVICE_NAME == 'npu' else 'nccl'


def _init_dist_process(rank, world_size, fn, backend, device_name, file_name):
    """Per-rank entry: bind the rank device and rendezvous on a FileStore.

    The parent process creates the temp file and passes the path; no
    MASTER_ADDR/MASTER_PORT is involved.
    """
    if device_name == 'npu':
        torch.npu.set_device(rank)
    else:
        torch.cuda.set_device(rank)
    store = dist.FileStore(file_name, world_size)
    dist.init_process_group(backend, rank=rank, world_size=world_size, store=store)
    try:
        fn(rank, world_size, device_name)
    finally:
        dist.destroy_process_group()


def _build_ddp(rank, device_name):
    # Deterministic init; DDP broadcasts rank0 parameters to all ranks.
    torch.manual_seed(7)
    device = f'{device_name}:{rank}'
    model = torch.nn.Linear(4, 4).float().to(device)
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])


def _scenario_disabled_false(rank, world_size, device_name):
    # With manual finalization disabled the property is False at all times.
    device = f'{device_name}:{rank}'
    ddp = _build_ddp(rank, device_name)
    assert ddp.should_finalize_after_backward is False, "disabled manual mode must report False"
    inp = torch.ones(8, 4).to(device)
    out = ddp(inp)
    assert ddp.should_finalize_after_backward is False, "disabled mode stays False after forward"
    out.sum().backward()
    assert ddp.should_finalize_after_backward is False, "disabled mode stays False after backward"
    assert isinstance(ddp.should_finalize_after_backward, bool), "property must return bool"


def _scenario_rebuild_true(rank, world_size, device_name):
    # With manual finalization enabled, the first iteration is inside bucket
    # rebuilding: True before forward and still True after backward + finalize.
    device = f'{device_name}:{rank}'
    ddp = _build_ddp(rank, device_name)
    ddp.require_manual_backward_finalization = True
    assert ddp.should_finalize_after_backward is True, "rebuilding phase must report True"
    inp = (rank + 1) * torch.ones(8, 4).to(device)
    out = ddp(inp)
    assert ddp.should_finalize_after_backward is True, "rebuilding phase must report True after forward"
    out.sum().backward()
    assert ddp.should_finalize_after_backward is True, "rebuilding phase must report True after backward"
    ddp.finalize_backward()
    assert ddp.should_finalize_after_backward is True, "rebuilding phase ends after first finalize"


def _scenario_rebuild_completed_false(rank, world_size, device_name):
    # After the rebuild completes, subsequent iterations report False so the
    # caller may finalize from a bucket-ready hook instead of after backward.
    device = f'{device_name}:{rank}'
    ddp = _build_ddp(rank, device_name)
    ddp.require_manual_backward_finalization = True
    inp = (rank + 1) * torch.ones(8, 4).to(device)
    out = ddp(inp)
    out.sum().backward()
    ddp.finalize_backward()
    ddp.zero_grad(set_to_none=True)
    out2 = ddp(inp)
    assert ddp.should_finalize_after_backward is False, "rebuilt buckets must report False after forward"
    out2.sum().backward()
    ddp.finalize_backward()
    grad = ddp.module.weight.grad
    assert grad is not None, "finalized iteration must leave grads written back"
    assert torch.allclose(grad.float().cpu(), torch.full_like(grad.float().cpu(), 12.0)), \
        f"rank {rank}: finalized grad must equal the two-rank average 12.0"


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDistributedDataParallelShouldFinalizeAfterBackward(TestCase):
    """Test cases for DDP.should_finalize_after_backward (multi-card HCCL)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def _spawn(self, scenario):
        world_size = 2
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_name = f.name
        try:
            mp.spawn(
                _init_dist_process,
                args=(world_size, scenario, BACKEND, self.device_name, file_name),
                nprocs=world_size,
                join=True,
            )
        finally:
            if os.path.exists(file_name):
                os.unlink(file_name)

    @skipIfUnsupportMultiNPU(2)
    def test_should_finalize_after_backward_false_when_disabled(self):
        # Disabled manual mode reports False before forward and after backward.
        self._spawn(_scenario_disabled_false)

    @skipIfUnsupportMultiNPU(2)
    def test_should_finalize_after_backward_true_during_bucket_rebuild(self):
        # First manual iteration stays True through backward and finalize.
        self._spawn(_scenario_rebuild_true)

    @skipIfUnsupportMultiNPU(2)
    def test_should_finalize_after_backward_false_after_rebuild_completed(self):
        # Later manual iterations report False at the caller decision point
        # (after forward, before backward); finalize still writes back the
        # averaged gradients.
        self._spawn(_scenario_rebuild_completed_false)


if __name__ == "__main__":
    run_tests()
