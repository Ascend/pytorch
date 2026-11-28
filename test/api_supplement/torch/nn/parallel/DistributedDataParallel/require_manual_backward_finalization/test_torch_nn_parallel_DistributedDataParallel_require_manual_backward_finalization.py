# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.parallel.DistributedDataParallel.require_manual_backward_finalization 接口功能正确性
API 名称：torch.nn.parallel.DistributedDataParallel.require_manual_backward_finalization
API 签名：require_manual_backward_finalization(self) -> bool（property，setter 接受 bool）
         （为 True 时 reducer 不在梯度桶就绪后自动收尾，调用方须显式 finalize_backward）

覆盖维度表：
| 覆盖维度         | 说明                                                  | 覆盖情况                                        |
|------------------|-------------------------------------------------------|-------------------------------------------------|
| 多卡集合通信     | world_size=2，HCCL 后端真实梯度 all-reduce            | 已覆盖（三个多卡用例）                          |
| 参数类型         | property 返回 bool；setter 接受 True/False            | 已覆盖（default_and_setter_roundtrip）          |
| 传参与不传参     | 默认 False；显式置 True / 置回 False                  | 已覆盖（default_and_setter_roundtrip）          |
| 等价类/边界值    | 布尔两态往返与重复读写                                | 已覆盖（default_and_setter_roundtrip）          |
| 正常传参场景     | True 时 backward 后梯度保持本地值（未自动收尾）；False 恢复自动写回平均值 | 已覆盖（true_defers / false_restores） |
| 异常传参场景     | 不适用：setter 对合法 bool 无异常路径（配置类拒绝属 _check 分支，本批未构造） | 不适用 |
| 混合设备类型     | 不适用：单模型单组参数，无多 Tensor 混合设备入参      | 不适用                                          |

未覆盖项及原因：
- delay_all_reduce_named_params / Python reducer 下置 True 的 RuntimeError
  拒绝路径：未覆盖。属 _check_manual_backward_finalization_supported 的
  配置类分支，构造该配置需专用参数组合，本批未构造。
- 序列化：未覆盖。__setstate__ 恢复默认 False 的行为属 DDP 整体序列化
  契约，非本 property 单点看护范围。
- 内存：不适用。开关属性无分配语义。

注意：多卡用例在 NPU 环境运行（HCCL），各 rank 输入按 rank 放大
     （(rank+1)*ones）使本地梯度 8*(rank+1) 与平均梯度 12.0 可区分：
     置 True 后 backward 直接读到的 .grad 必须是本地值（未自动收尾），
     置回 False 后 backward 直接写回平均值；通信结果以各 rank 的期望
     数值为 Golden；device_count < 2 时以 skipTest 归入环境不具备。
     开关语义为实测固定契约（torch_npu 2.15.0.dev20261118 + 2.15.0 源码）。
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


def _scenario_default_and_setter_roundtrip(rank, world_size, device_name):
    # Fresh DDP defaults to False; setter toggles both ways and is stable.
    ddp = _build_ddp(rank, device_name)
    assert ddp.require_manual_backward_finalization is False, "fresh DDP must default to False"
    ddp.require_manual_backward_finalization = True
    assert ddp.require_manual_backward_finalization is True, "setter True must be observable"
    ddp.require_manual_backward_finalization = False
    assert ddp.require_manual_backward_finalization is False, "setter False must be observable"
    for required in (True, False):
        ddp.require_manual_backward_finalization = required
        assert ddp.require_manual_backward_finalization is required, "repeated toggling must stay stable"
        assert isinstance(ddp.require_manual_backward_finalization, bool), "property must return bool"


def _scenario_true_defers_finalization(rank, world_size, device_name):
    # With True, backward must NOT auto-finalize: the observable .grad equals
    # the LOCAL value 8*(rank+1), not the all-reduced average 12.0.
    device = f'{device_name}:{rank}'
    ddp = _build_ddp(rank, device_name)
    ddp.require_manual_backward_finalization = True
    inp = (rank + 1) * torch.ones(8, 4).to(device)
    out = ddp(inp)
    out.sum().backward()
    grad = ddp.module.weight.grad
    assert grad is not None, "backward must leave grad accumulations present"
    local = torch.full_like(grad.float().cpu(), 8.0 * (rank + 1))
    assert torch.allclose(grad.float().cpu(), local), \
        f"rank {rank}: with manual finalization the pre-finalize grad must stay local (8*(rank+1))"
    # Complete this iteration's manual finalization. The averaged value after
    # finalize is asserted by the dedicated finalize_backward cases.
    ddp.finalize_backward()


def _scenario_false_restores_auto(rank, world_size, device_name):
    # Setting False restores automatic finalization: backward directly writes
    # the all-reduced average 12.0 with no finalize_backward call.
    device = f'{device_name}:{rank}'
    ddp = _build_ddp(rank, device_name)
    ddp.require_manual_backward_finalization = True
    ddp.require_manual_backward_finalization = False
    assert ddp.require_manual_backward_finalization is False
    inp = (rank + 1) * torch.ones(8, 4).to(device)
    out = ddp(inp)
    out.sum().backward()
    grad = ddp.module.weight.grad
    assert grad is not None, "auto backward must write grads"
    averaged = torch.full_like(grad.float().cpu(), 12.0)
    assert torch.allclose(grad.float().cpu(), averaged), \
        f"rank {rank}: auto-mode grad must equal the two-rank average 12.0"


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDistributedDataParallelRequireManualBackwardFinalization(TestCase):
    """Test cases for DDP.require_manual_backward_finalization (multi-card HCCL)."""

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
    def test_require_manual_backward_finalization_default_and_setter_roundtrip(self):
        # Default False, both setter directions observable, toggling stable.
        self._spawn(_scenario_default_and_setter_roundtrip)

    @skipIfUnsupportMultiNPU(2)
    def test_require_manual_backward_finalization_true_defers_finalization(self):
        # True keeps pre-finalize grads local (no automatic write-back).
        self._spawn(_scenario_true_defers_finalization)

    @skipIfUnsupportMultiNPU(2)
    def test_require_manual_backward_finalization_false_restores_auto_backward(self):
        # False restores automatic gradient write-back after backward.
        self._spawn(_scenario_false_restores_auto)


if __name__ == "__main__":
    run_tests()
