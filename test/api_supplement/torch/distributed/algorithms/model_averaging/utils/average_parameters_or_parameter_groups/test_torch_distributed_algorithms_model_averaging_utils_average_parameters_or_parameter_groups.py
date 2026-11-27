# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.algorithms.model_averaging.utils.average_parameters_or_parameter_groups 接口功能正确性
API 名称：torch.distributed.algorithms.model_averaging.utils.average_parameters_or_parameter_groups
API 签名：average_parameters_or_parameter_groups(params, process_group)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|-------------------------------------|--------------------------------------------------|
| 多卡集合通信   | world_size=2，HCCL 后端真实参数平均 | 已覆盖（test_model_parameters、test_optimizer_param_groups） |
| 参数类型       | Iterable[Parameter] 与 param_groups | 已覆盖（test_model_parameters、test_optimizer_param_groups） |
| 传参与不传参   | process_group=None 与显式进程组/子组 | 已覆盖（test_model_parameters、test_optimizer_param_groups） |
| 结果数值对照   | 有 grad 参数取两卡均值，无 grad 参数保持原值 | 已覆盖（test_model_parameters、test_optimizer_param_groups） |
| 进程组生命周期 | FileStore 临时文件 init/destroy_process_group | 已覆盖（全部多卡用例） |
| 异常路径       | 无任何 grad/空输入 RuntimeError、非法类型 NotImplementedError | 已覆盖（test_invalid_inputs_raises） |

未覆盖项及原因：
- 无

注意：多卡用例在 2 卡 NPU/CUDA 环境通过 HCCL/NCCL 运行，验证仅对 grad 非空的参数执行跨卡平均，
     grad 为 None 的参数在各 rank 保持各自原始值不变。
"""
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.algorithms.model_averaging.utils import average_parameters_or_parameter_groups

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


def _worker_model_parameters(rank, world_size, device_name):
    local_device = f"{device_name}:{rank}"
    p_with_grad = nn.Parameter(torch.full((4, 4), float(2 + 4 * rank), device=local_device))
    p_without_grad = nn.Parameter(torch.full((4, 4), float(10 + 5 * rank), device=local_device))
    p_with_grad.grad = torch.ones_like(p_with_grad)
    average_parameters_or_parameter_groups([p_with_grad, p_without_grad], process_group=None)
    # p_with_grad: (2+6)/2=4.0; p_without_grad untouched.
    torch.testing.assert_close(p_with_grad.data, torch.full((4, 4), 4.0, device=local_device))
    torch.testing.assert_close(p_without_grad.data, torch.full((4, 4), float(10 + 5 * rank), device=local_device))


def _worker_optimizer_param_groups(rank, world_size, device_name):
    local_device = f"{device_name}:{rank}"
    sub_pg = dist.new_group(ranks=list(range(world_size)))
    layer1 = nn.Linear(4, 3, bias=True).to(local_device)
    layer2 = nn.Linear(3, 2, bias=True).to(local_device)
    with torch.no_grad():
        layer1.weight.fill_(float(1 + 2 * rank))
        layer1.bias.fill_(float(10 + 2 * rank))
        layer2.weight.fill_(float(4 + 6 * rank))
        layer2.bias.fill_(float(20 + 4 * rank))
    layer1.weight.grad = torch.ones_like(layer1.weight)
    layer2.weight.grad = torch.ones_like(layer2.weight)
    optimizer = torch.optim.SGD([
        {"params": list(layer1.parameters()), "lr": 0.01},
        {"params": list(layer2.parameters()), "lr": 0.05},
    ])
    average_parameters_or_parameter_groups(optimizer.param_groups, process_group=sub_pg)
    torch.testing.assert_close(layer1.weight.data, torch.full((3, 4), 2.0, device=local_device))
    torch.testing.assert_close(layer2.weight.data, torch.full((2, 3), 7.0, device=local_device))
    torch.testing.assert_close(layer1.bias.data, torch.full((3,), float(10 + 2 * rank), device=local_device))
    torch.testing.assert_close(layer2.bias.data, torch.full((2,), float(20 + 4 * rank), device=local_device))


def _worker_invalid_inputs_raises(rank, world_size, device_name):
    dev = f"{device_name}:{rank}"
    p_no_grad = nn.Parameter(torch.ones(4, device=dev))
    _assert_raises(
        (RuntimeError, ValueError), average_parameters_or_parameter_groups, [p_no_grad], process_group=dist.group.WORLD
    )
    _assert_raises(
        (RuntimeError, ValueError), average_parameters_or_parameter_groups, [], process_group=dist.group.WORLD
    )
    plain_tensor = torch.ones(4, device=dev)
    _assert_raises(
        NotImplementedError, average_parameters_or_parameter_groups, [plain_tensor], process_group=dist.group.WORLD
    )


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestAverageParametersOrParameterGroups(TestCase):
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
    def test_model_parameters(self):
        self._run_mp(_worker_model_parameters)

    @skipIfUnsupportMultiNPU(2)
    def test_optimizer_param_groups(self):
        self._run_mp(_worker_optimizer_param_groups)

    @skipIfUnsupportMultiNPU(2)
    def test_invalid_inputs_raises(self):
        self._run_mp(_worker_invalid_inputs_raises)


if __name__ == "__main__":
    run_tests()
