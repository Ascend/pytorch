# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.parallel.DistributedDataParallel.finalize_backward 接口功能正确性
API 名称：torch.nn.parallel.DistributedDataParallel.finalize_backward
API 签名：finalize_backward(self) -> None
         （等待在途梯度 all-reduce 完成并把归约后梯度写回参数 .grad）

覆盖维度表：
| 覆盖维度         | 说明                                                  | 覆盖情况                                        |
|------------------|-------------------------------------------------------|-------------------------------------------------|
| 多卡集合通信     | world_size=2，HCCL 后端真实梯度 all-reduce 与写回     | 已覆盖（write_back_reduced_grads）              |
| 同步/异步分支    | 手动 finalization：backward 后统一 finalize 收尾      | 已覆盖（write_back_reduced_grads）              |
| 结果数值对照     | finalize 后各 rank 梯度等于归约平均值（weight 12.0，bias 梯度与 rank 无关恒 8.0） | 已覆盖（write_back_reduced_grads）   |
| 进程组生命周期   | init / destroy_process_group 正常                     | 已覆盖（全部多卡用例）                          |
| rendezvous       | FileStore 临时文件（无 MASTER_PORT）                  | 已覆盖（全部多卡用例）                          |
| 异常路径         | 未开启手动模式调用报 RuntimeError；同一 backward 重复 finalize 报 RuntimeError | 已覆盖（error_when_not_required / consecutive） |

未覆盖项及原因：
- no_sync() 期间 finalize：未覆盖。no_sync 场景梯度不同步，与本 API
  手动 finalization 契约的组合行为缺稳定可复现入口，不凭空断言。
- delay_all_reduce_named_params / Python reducer 下的拒绝路径：未覆盖。
  属 _check_manual_backward_finalization_supported 的配置类分支，构造
  该配置需专用参数组合，本批未构造。
- 内存：未覆盖。finalize_backward 为同步收尾动作，无独立分配语义可看护。

注意：多卡用例在 NPU 环境运行（HCCL），各 rank 输入按 rank 放大
     （(rank+1)*ones）使本地梯度 8*(rank+1) 与平均梯度 12.0 可区分，
     finalize 后梯度必须等于平均值，非平均值即失败；通信结果以各 rank
     的期望数值为 Golden；device_count < 2 时以 skipTest 归入环境不具备，
     不伪造对照。异常消息为实测固定文本。
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


def _scenario_finalize_write_back(rank, world_size, device_name):
    # Rank-distinct inputs make the local weight gradient 8*(rank+1) differ
    # from the all-reduced average 12.0, so the write-back contract is
    # falsifiable via the weight. The bias gradient is rank-independent
    # (grad_out is all ones), so its averaged value stays 8.0 on every rank.
    device = f'{device_name}:{rank}'
    ddp = _build_ddp(rank, device_name)
    ddp.require_manual_backward_finalization = True
    inp = (rank + 1) * torch.ones(8, 4).to(device)
    out = ddp(inp)
    out.sum().backward()
    weight = ddp.module.weight
    bias = ddp.module.bias
    assert weight.grad is not None, "backward must leave grad accumulations present"
    ddp.finalize_backward()
    assert torch.allclose(weight.grad.float().cpu(), torch.full_like(weight.grad.float().cpu(), 12.0)), \
        f"rank {rank}: finalized weight grad must equal the two-rank average 12.0"
    assert torch.allclose(bias.grad.float().cpu(), torch.full_like(bias.grad.float().cpu(), 8.0)), \
        f"rank {rank}: finalized bias grad must equal the rank-independent average 8.0"


def _scenario_finalize_error_not_required(rank, world_size, device_name):
    # Calling finalize_backward without manual finalization raises RuntimeError
    # with the measured fixed message.
    ddp = _build_ddp(rank, device_name)
    try:
        ddp.finalize_backward()
        raise AssertionError("finalize_backward() without manual mode must raise RuntimeError")
    except RuntimeError as exc:
        assert "manual backward finalization is not required" in str(exc), \
            f"unexpected RuntimeError text: {exc}"


def _scenario_finalize_consecutive_iterations(rank, world_size, device_name):
    # Two consecutive manual iterations each finalize exactly once; a second
    # finalize for the same backward raises the measured RuntimeError.
    device = f'{device_name}:{rank}'
    ddp = _build_ddp(rank, device_name)
    ddp.require_manual_backward_finalization = True
    inp = (rank + 1) * torch.ones(8, 4).to(device)
    for _ in range(2):
        ddp.zero_grad(set_to_none=True)
        out = ddp(inp)
        out.sum().backward()
        ddp.finalize_backward()
        grad = ddp.module.weight.grad
        assert grad is not None, "finalized iteration must leave grads written back"
        assert torch.allclose(grad.float().cpu(), torch.full_like(grad.float().cpu(), 12.0)), \
            f"rank {rank}: finalized grad must equal the two-rank average 12.0"
        try:
            ddp.finalize_backward()
            raise AssertionError("a second finalize for the same backward must raise RuntimeError")
        except RuntimeError as exc:
            assert "no gradient reduction requires finalization" in str(exc), \
                f"unexpected RuntimeError text: {exc}"


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDistributedDataParallelFinalizeBackward(TestCase):
    """Test cases for DistributedDataParallel.finalize_backward (multi-card HCCL)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def _spawn(self, scenario):
        world_size = 2
        if self.device_name == 'cpu':
            self.skipTest("multi-device scenario requires npu or cuda devices")
        device_count = (
            torch.npu.device_count() if self.device_name == 'npu' else torch.cuda.device_count()
        )
        if device_count < world_size:
            self.skipTest(
                f"{self.device_name} device count {device_count} < world_size {world_size}"
            )
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
    def test_finalize_backward_write_back_reduced_grads(self):
        # finalize_backward waits for in-flight all-reduce and writes the
        # averaged gradients back into .grad on every rank.
        self._spawn(_scenario_finalize_write_back)

    @skipIfUnsupportMultiNPU(2)
    def test_finalize_backward_error_when_not_required(self):
        # RuntimeError when require_manual_backward_finalization is False.
        self._spawn(_scenario_finalize_error_not_required)

    @skipIfUnsupportMultiNPU(2)
    def test_finalize_backward_consecutive_iterations_and_single_use(self):
        # Sequential manual iterations work; finalize is single-use per
        # backward pass.
        self._spawn(_scenario_finalize_consecutive_iterations)


if __name__ == "__main__":
    run_tests()
