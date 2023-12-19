import copy
import functools

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils.checkpoint import checkpoint
from functorch.compile import min_cut_rematerialization_partition
from torch._dynamo.backends.common import aot_autograd

import torch_npu
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU


class SimpleModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mlp_0 = MLPModule(device)
        self.mlp_1 = MLPModule(device)

    def forward(self, input):
        return self.mlp_1(self.mlp_0(input))


def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return fx_g


# Make a custom compiler that runs aot autograd but extracts the fw graph
fw_graph_cell = [None]
bw_graph_cell = [None]
fw_compiler = functools.partial(extract_graph, graph_cell=fw_graph_cell)
bw_compiler = functools.partial(extract_graph, graph_cell=bw_graph_cell)

aot_eager_graph = aot_autograd(
    fw_compiler=fw_compiler,
    bw_compiler=bw_compiler,
    partition_fn=min_cut_rematerialization_partition,
)


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

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=False)
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

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=False)
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
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=False)
        res = opt_fn(x)
        self.assertEqual(res, ref)

        # test if user calls from_local with mesh/placements as kwargs and that should still work
        def from_local_kwargs_fn(x):
            dt = DTensor.from_local(
                x, device_mesh=mesh, placements=[Replicate()], run_check=False
            )
            return dt.to_local() + 2

        ref = from_local_kwargs_fn(x)
        opt_kwargs_fn = torch.compile(
            from_local_kwargs_fn, backend="eager", fullgraph=True, dynamic=False
        )
        res = opt_kwargs_fn(x)
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
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=False)
        res = opt_fn(x)
        self.assertEqual(res, ref)

        def redistribute_kwargs_fn(x):
            dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
            return (
                dt.redistribute(device_mesh=mesh, placements=[Replicate()]).to_local()
                + 2
            )

        x = torch.ones(1)
        ref = redistribute_kwargs_fn(x)
        opt_kwargs_fn = torch.compile(
            redistribute_kwargs_fn, backend="eager", fullgraph=True, dynamic=False
        )
        res = opt_kwargs_fn(x)
        self.assertEqual(res, ref)


class TestDTensorCompileE2E(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    @parametrize("is_seq_parallel", [True, False])
    def test_tp_compile_fullgraph(self, is_seq_parallel):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model = SimpleModel(self.device_type)
        module_prepare_input = (
            PrepareModuleInput()
            if is_seq_parallel
            else PrepareModuleInput(input_layouts=Replicate())
        )
        no_input_prepare_colwise_style = ColwiseParallel(input_layouts=None)
        colwise_style = (
            ColwiseParallel(input_layouts=Shard(0))
            if is_seq_parallel
            else ColwiseParallel()
        )
        rowwise_style = (
            RowwiseParallel(output_layouts=Shard(0))
            if is_seq_parallel
            else RowwiseParallel()
        )
        model = parallelize_module(
            model,
            mesh,
            parallelize_plan={
                "mlp_0": module_prepare_input,
                "mlp_0.net1": no_input_prepare_colwise_style,
                "mlp_0.net2": rowwise_style,
                "mlp_1.net1": colwise_style,
                "mlp_1.net2": rowwise_style,
            },
        )
        rng_seed = self.rank if is_seq_parallel else 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(20, 10, device=self.device_type)
        out = model(inp)
        compiled_mod = torch.compile(
            model, backend="aot_eager", fullgraph=True, dynamic=False
        )
        compiled_out = compiled_mod(inp)
        self.assertEqual(compiled_out, out)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_2d_fsdp_tp_compile(self):
        data_parallel_size = 2
        model = SimpleModel(self.device_type)
        model_copy = copy.deepcopy(model)

        # 2-D mesh is [dp, tp]
        twod_mesh = init_device_mesh(
            "npu",
            (data_parallel_size, self.world_size // data_parallel_size),
            mesh_dim_names=["dp", "tp"],
        )

        fsdp_pg = twod_mesh.get_dim_groups()[0]

        inp = torch.rand(20, 10, device=self.device_type)
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, twod_mesh["tp"], parallelize_plan)
        eager_2d = FSDP(
            tp_model,
            device_id=self.rank,
            use_orig_params=True,
            device_mesh=twod_mesh["dp"],
        )
        out = eager_2d(inp)
        tp_model2 = parallelize_module(
            model_copy,
            twod_mesh["tp"],
            parallelize_plan,
        )
        fsdp_2d = FSDP(
            tp_model2,
            device_id=self.rank,
            use_orig_params=True,
            device_mesh=twod_mesh["dp"],
        )

        compiled_2d = torch.compile(fsdp_2d, backend="aot_eager", dynamic=False)
        compiled_output = compiled_2d(inp)

        self.assertEqual(out, compiled_output)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_2d_fsdp_tp_ac_compile(self):
        dp_degree = 2
        tp_degree = self.world_size // dp_degree
        model = SimpleModel(self.device_type)
        model_copy = copy.deepcopy(model)

        # 2-D mesh is [dp, tp]
        mesh_2d = init_device_mesh(
            "npu", mesh_shape=(dp_degree, tp_degree), mesh_dim_names=("dp", "tp")
        )

        inp = torch.rand(20, 10, device=self.device_type)
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, mesh_2d["tp"], parallelize_plan)
        tp_model = checkpoint_wrapper(
            tp_model,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            use_reentrant=False,
        )
        eager_2d = FSDP(tp_model, device_mesh=mesh_2d["dp"], use_orig_params=True)

        tp_model2 = parallelize_module(model_copy, mesh_2d["tp"], parallelize_plan)
        fsdp_2d = FSDP(
            tp_model2,
            device_mesh=mesh_2d["dp"],
            use_orig_params=True,
        )

        compiled_2d = torch.compile(fsdp_2d, backend="aot_eager", dynamic=False)

        # forward pass
        out = eager_2d(inp)
        compiled_output = compiled_2d(inp)
        self.assertEqual(out, compiled_output)

        # backward pass
        out.sum().backward()
        compiled_output.sum().backward()

        # compare the gradients:
        for n, p in zip(fsdp_2d.parameters(), compiled_2d.parameters()):
            self.assertEqual(n.grad, p.grad)

instantiate_parametrized_tests(TestDTensorCompileE2E)

if __name__ == "__main__":
    run_tests()
