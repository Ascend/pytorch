# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.optimizer.load_sharded_optimizer_state_dict 接口功能正确性
API 名称：torch.distributed.checkpoint.optimizer.load_sharded_optimizer_state_dict
API 签名：load_sharded_optimizer_state_dict(model_state_dict, optimizer_key, storage_reader, planner=None)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 多卡集合通信 | world_size=2，HCCL 真实保存与加载 | 已覆盖（test_save_and_load_roundtrip） |
| 同步/异步分支 | 接口为同步加载，无 async_op 分支 | 未覆盖（API 本身无异步参数） |
| 结果数值对照 | 分片状态、标量 step 与 param_groups 对照 | 已覆盖（test_save_and_load_roundtrip） |
| 进程组生命周期 | init_process_group / destroy_process_group 完整生命周期 | 已覆盖（test_save_and_load_roundtrip） |
| rendezvous | FileStore 临时文件（无 MASTER_PORT） | 已覆盖（test_save_and_load_roundtrip） |
| 参数传参与边界 | 默认/显式 planner、缺失 optimizer_key | 已覆盖（test_save_and_load_roundtrip） |
| 异常路径 | 缺失元数据目录抛 FileNotFoundError、缺参抛 TypeError | 已覆盖（test_missing_checkpoint_and_args_raises） |

未覆盖项及原因：
- 同步/异步分支：load_sharded_optimizer_state_dict 仅提供同步接口，不支持 async_op。

注意：多卡用例在 NPU/CUDA 环境通过 mp.spawn + FileStore 临时文件 rendezvous 运行；
     各 rank 恢复的 ShardedTensor 分片与标量 Tensor 移至 CPU 后以 torch.testing.assert_close
     与保存前的原始优化器状态数值逐片对照；设备数不足 2 时以 self.skipTest 归入环境不具备。
"""
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict

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


def _build_and_step_model_optimizer(device):
    torch.manual_seed(1234)
    model = torch.nn.Linear(4, 4, bias=True, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=0.05)
    model(torch.arange(8, dtype=torch.float32, device=device).view(2, 4)).sum().backward()
    optim.step()
    return get_model_state_dict(model), get_optimizer_state_dict(model, optim)


def _verify_loaded_optimizer_state(loaded, orig_optim_sd, optimizer_key):
    assert optimizer_key in loaded, f"Missing {optimizer_key} in loaded state_dict"
    assert "model" not in loaded, "Non-optimizer keys must not be loaded"
    restored_opt = loaded[optimizer_key]
    assert restored_opt["param_groups"] == orig_optim_sd["param_groups"], "Restored param_groups mismatch"
    assert set(restored_opt["state"].keys()) == set(orig_optim_sd["state"].keys()), (
        "Restored optimizer state param IDs mismatch"
    )
    for param_id, orig_state in orig_optim_sd["state"].items():
        restored_state = restored_opt["state"][param_id]
        for key, orig_val in orig_state.items():
            loaded_val = restored_state[key]
            if isinstance(loaded_val, ShardedTensor):
                local_shards = loaded_val.local_shards()
                assert len(local_shards) > 0, f"Expected local shards for {param_id}.{key}"
                for shard in local_shards:
                    slices = tuple(
                        slice(o, o + s) for o, s in zip(shard.metadata.shard_offsets, shard.metadata.shard_sizes)
                    )
                    torch.testing.assert_close(shard.tensor.cpu(), orig_val[slices].cpu())
            else:
                torch.testing.assert_close(loaded_val.cpu(), orig_val.cpu())


def _worker_save_and_load(rank, world_size, device_name, ckpt_dir):
    opt_key = "optimizer"
    model_sd, orig_optim_sd = _build_and_step_model_optimizer(torch.device(f"{device_name}:{rank}"))
    dcp.save(state_dict={"model": model_sd, opt_key: orig_optim_sd}, storage_writer=FileSystemWriter(ckpt_dir))
    dist.barrier()

    loaded_default = load_sharded_optimizer_state_dict(
        model_state_dict=model_sd,
        optimizer_key=opt_key,
        storage_reader=FileSystemReader(ckpt_dir),
    )
    _verify_loaded_optimizer_state(loaded_default, orig_optim_sd, opt_key)

    loaded_explicit = load_sharded_optimizer_state_dict(
        model_state_dict=model_sd,
        optimizer_key=opt_key,
        storage_reader=FileSystemReader(ckpt_dir),
        planner=DefaultLoadPlanner(),
    )
    _verify_loaded_optimizer_state(loaded_explicit, orig_optim_sd, opt_key)

    empty_loaded = load_sharded_optimizer_state_dict(
        model_state_dict=model_sd,
        optimizer_key="nonexistent_optimizer",
        storage_reader=FileSystemReader(ckpt_dir),
    )
    assert empty_loaded == {}, f"Expected empty dict for missing key, got {empty_loaded}"


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLoadShardedOptimizerStateDict(TestCase):
    def _run_mp(self, fn, world_size=2, *extra):
        count = (torch.npu if DEVICE_NAME == "npu" else torch.cuda).device_count()
        if count < world_size:
            self.skipTest(f"Requires at least {world_size} {DEVICE_NAME} devices, got {count}")
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
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            self._run_mp(_worker_save_and_load, 2, ckpt_dir)

    def test_missing_checkpoint_and_args_raises(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(FileNotFoundError):
                load_sharded_optimizer_state_dict(
                    model_state_dict={},
                    optimizer_key="optimizer",
                    storage_reader=FileSystemReader(empty_dir),
                )
        with self.assertRaises(TypeError):
            load_sharded_optimizer_state_dict()


if __name__ == "__main__":
    run_tests()
