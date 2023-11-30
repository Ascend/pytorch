import copy
import os

import torch
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestDdpCommHook(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.randn(40, 20))

    def forward(self, x):
        return self.p * x


class HcomAllReduceTest(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        return dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)

    @classmethod
    def _get_grad(cls, model, train_data):
        output = model(train_data)
        output.mean().backward()
        param = next(model.parameters())
        return param.grad

    @classmethod
    def _test_hook(cls, rank, world_size, hook_type):
        torch.npu.manual_seed(0)
        torch.manual_seed(0)
        pg = HcomAllReduceTest._init_dist_hccl(rank, world_size)
        torch.npu.set_device(rank)
        origin_model = TestDdpCommHook()
        train_data = torch.randn(40, 20).npu()
        no_hook_model = nn.parallel.DistributedDataParallel(copy.deepcopy(origin_model).npu(), device_ids=[rank])
        no_hook_grad = HcomAllReduceTest._get_grad(no_hook_model, train_data)

        hook_model = nn.parallel.DistributedDataParallel(copy.deepcopy(origin_model).npu(), device_ids=[rank])
        hook_model.register_comm_hook(state=pg, hook=hook_type)
        hook_grad = HcomAllReduceTest._get_grad(hook_model, train_data)
        TestCase().assertEqual(hook_grad, no_hook_grad)

    @skipIfUnsupportMultiNPU(2)
    def test_fp16_compress_hook(self):
        # CI currently supports only 2 devices
        world_size = 2
        mp.spawn(HcomAllReduceTest._test_hook,
                 args=(world_size, default_hooks.fp16_compress_hook,),
                 nprocs=world_size,
                 join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_allreduce_hook(self):
        # CI currently supports only 2 devices
        world_size = 2
        mp.spawn(HcomAllReduceTest._test_hook,
                 args=(world_size, default_hooks.allreduce_hook,),
                 nprocs=world_size,
                 join=True)


if __name__ == '__main__':
    run_tests()
