import os
from copy import deepcopy

import torch

import torch.distributed as dist
from torch import nn
import torch.distributed
import torch.multiprocessing as mp
import torch.distributed._functional_collectives as fcol

from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import normalize_gm

import torch_npu


DIM = 256


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM, DIM)

    def forward(self, x, world_size):
        _fc1 = self.fc1(x)
        torch.distributed.all_reduce(_fc1)
        _fc2 = _fc1.new_zeros([_fc1.shape[0] // world_size, *_fc1.shape[1:]])
        torch.distributed.reduce_scatter_tensor(_fc2, _fc1)
        _fc3 = _fc2.new_zeros([_fc2.shape[0] * world_size, *_fc2.shape[1:]])
        torch.distributed.all_gather_into_tensor(_fc3, _fc2)
        torch.distributed.all_to_all_single(_fc3, _fc3)
        _fc3 = _fc3.reshape(2, -1)[0]
        return _fc3


def _test_compile(
    rank,
    world_size,
):
    backend = "hccl"
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    graph = None

    def compiler_fn(gm):
        def inner_compiler(gm_, example_inputs_):
            nonlocal graph
            if graph is not None:
                raise AssertionError('TestCommConverter Failed, before run, graph should be None')
            graph = gm_
            graph = normalize_gm(graph.print_readable(False))
            import torchair
            return torchair.get_npu_backend()(gm_, example_inputs_)

        return torch.compile(
            gm, backend=inner_compiler, dynamic=False, fullgraph=True
        )

    torch_npu.npu.set_device(f"npu:{rank}")
    device = torch.device("npu")
    torch.manual_seed(123)
    model = Net().to(device)
    expect = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        _fc1 = self.L__self___fc1(l_x_);  l_x_ = None

        tensor = torch.ops.c10d_functional.all_reduce(_fc1, 'sum', 'ptd:0', [0], 1)

        wait_tensor = torch.ops.c10d_functional.wait_tensor(tensor);  tensor = None

        copy_ = _fc1.copy_(wait_tensor);  wait_tensor = None

        _fc2 = _fc1.new_zeros([256, 256])

        tensor_1 = torch.ops.c10d_functional.reduce_scatter_tensor(_fc1, 'sum', 'ptd:0', [0], 1);  _fc1 = None

        res = torch.ops.c10d_functional.wait_tensor(tensor_1);  tensor_1 = None

        copy__1 = _fc2.copy_(res);  res = None

        _fc3 = _fc2.new_zeros([256, 256])

        tensor_2 = torch.ops.c10d_functional.all_gather_into_tensor(_fc2, 'ptd:0', [0], 1);  _fc2 = None

        res_1 = torch.ops.c10d_functional.wait_tensor(tensor_2);  tensor_2 = None

        copy__2 = _fc3.copy_(res_1);  res_1 = None

        tensor_3 = torch.ops.c10d_functional.all_to_all_single(_fc3, None, None, 'ptd:0', [0], 1)

        wait_tensor_3 = torch.ops.c10d_functional.wait_tensor(tensor_3);  tensor_3 = None

        copy__3 = _fc3.copy_(wait_tensor_3);  wait_tensor_3 = None

        reshape = _fc3.reshape(2, -1);  _fc3 = None
        _fc3_1 = reshape[0];  reshape = None
        return (_fc3_1,)
"""
    compiled_model = compiler_fn(deepcopy(model))
    ret = []
    for i in range(3):
        torch.manual_seed(123 + rank + i)
        input_tensor = torch.randn([DIM, DIM], device=device)
        input_tensor_copy = input_tensor.clone()
        compiled_output = compiled_model(input_tensor, world_size)
        loss_output = model(input_tensor_copy, world_size)
        if expect != graph:
            raise RuntimeError('TestCommConverter Failed, fx graph is not expected')
        if not torch.isclose(compiled_output, loss_output, rtol=1e-5, atol=1e-5).all():
            raise RuntimeError('TestCommConverter Failed, dynamo outputs are not equal to eager outputs')


def mp_main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['TORCH_DISABLE_NATIVE_FUNCOL'] = '1'
    _test_compile(rank=rank, world_size=world_size)


class TestCommConverter(TestCase):
    def test_comm_converter(self):
        world_size = 1
        mp.spawn(mp_main, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":

    from torch._dynamo.test_case import run_tests

    run_tests()
