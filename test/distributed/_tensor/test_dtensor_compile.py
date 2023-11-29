import copy

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule
)
from torch.testing._internal.distributed.fake_pg import FakeStore

import torch_npu
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import run_tests


class SimpleModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mlp_0 = MLPModule(device)
        self.mlp_1 = MLPModule(device)

    def forward(self, input_x):
        return self.mlp_1(self.mlp_0(input_x))


class TestDTensorCompile(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_fakify_dtensor(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # pass in DTensor as inputs/outputs to the function
        def fn(x):
            return x

        x = DTensor.from_local(torch.rand(1), mesh, [Shard(0)], run_check=False)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_dynamo_dtensor(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            return x * x + 2

        x = DTensor.from_local(torch.rand(1), mesh, [Shard(0)], run_check=False)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_dynamo_dtensor_from_local(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # create DTensor inside fn and run some compute
        def fn(x):
            dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)
            return dt.to_local() + 2

        x = torch.ones(1)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_dynamo_dtensor_from_local_redistribute(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # pass in tensor as inputs/outputs, create DTensor and run redistribute
        # (allgather collective) inside the fn
        def fn(x):
            dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
            return dt.redistribute(mesh, [Replicate()]).to_local() + 2

        x = torch.ones(1)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)


class TestDTensorCompileE2E(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_tp_compile_fullgraph(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model = MLPModule(self.device_type)
        model = parallelize_module(model, mesh, PairwiseParallel())
        inp = torch.rand(20, 10, device=self.device_type)
        out = model(inp)
        compiled_mod = torch.compile(model, backend="eager", fullgraph=True)
        compiled_out = compiled_mod(inp)
        self.assertEqual(compiled_out, out)

    @with_comms
    @skipIfUnsupportMultiNPU(4)
    def test_2d_fsdp_tp_compile(self):
        data_parallel_size = 2
        model = SimpleModel(self.device_type)
        model_copy = copy.deepcopy(model)
        enable_2d_with_fsdp()

        # 2-D mesh is [dp, tp]
        twod_mesh = DeviceMesh(
            device_type=self.device_type,
            mesh=torch.arange(0, self.world_size).view(data_parallel_size, -1),
        )

        fsdp_pg = twod_mesh.get_dim_groups()[0]

        inp = torch.rand(20, 10, device=self.device_type)
        tp_model = parallelize_module(
            model, twod_mesh, PairwiseParallel(), tp_mesh_dim=1
        )
        eager_2d = FSDP(
            tp_model, process_group=fsdp_pg, device_id=self.rank, use_orig_params=True
        )
        out = eager_2d(inp)
        tp_model2 = parallelize_module(
            model_copy, twod_mesh, PairwiseParallel(), tp_mesh_dim=1
        )
        compiled_tp = torch.compile(tp_model2, backend="eager", fullgraph=True)

        # we should apply torch.compile after fsdp wrap, but the current graph break approach
        # have some issues with the tensor subclass compilation, need to dig into this later
        compiled_2d = FSDP(
            compiled_tp,
            process_group=fsdp_pg,
            device_id=self.rank,
            use_orig_params=True,
        )

        compiled_output = compiled_2d(inp)

        self.assertEqual(out, compiled_output)


if __name__ == "__main__":
    run_tests()