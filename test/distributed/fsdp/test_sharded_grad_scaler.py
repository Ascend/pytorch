from functools import wraps
import os
import types

import torch
import torch.distributed as dist

import torch_npu
from torch_npu.npu.amp.sharded_grad_scaler import _ShardedGradScaler as NpuShardedGradScaler
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)


class TestShardedGradScaler(TestCase):
    MAIN_PROCESS_RANK = -1

    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                for p in self.processes:
                    p.join()
            else:
                fn()

        return types.MethodType(wrapper, self)

    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        fn = getattr(self, method_name)
        setattr(self, method_name, self.join_or_run(fn))

    def setUp(self):
        super(TestCase, self).setUp()
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29588'
        os.environ["BACKEND"] = dist.Backend.HCCL
        self.processes = []
        self.rank = self.MAIN_PROCESS_RANK
        proc = torch.multiprocessing.get_context("spawn").Process

        for rank in range(int(self.world_size)):
            process = proc(
                target=self.__class__._run,
                name="process " + str(rank),
                args=(rank, self._current_test_name()),
            )
            process.start()
            self.processes.append(process)

    def tearDown(self):
        super().tearDown()
        for p in self.processes:
            p.terminate()
        self.processes = []

    def _current_test_name(self) -> str:
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    @property
    def world_size(self) -> int:
        return 2

    @classmethod
    def _run(cls, rank: int, test_name: str) -> None:
        self = cls(test_name)
        self.rank = rank
        getattr(self, test_name)()

    def dist_init(self):
        torch.npu.set_device(self.rank)
        dist.init_process_group(backend="hccl", rank=self.rank, world_size=self.world_size)

    def float_close(self, a, b, atol=1e-3):
        return abs(a - b) < atol

    @skipIfUnsupportMultiNPU(2)
    def test_sharded_grad_scaler_npu(self):
        '''
        Testcase for non-offload FSDP sharded grad scaler.
        We init scale to 2 ** 16, and overflow every 2 steps. then check if the scale is updated correctly.
        '''
        self.dist_init()
        device = f"npu:{self.rank}"
        model = MyModule().to(device)
        # Set the update interval to 1 to make the scale change more frequently.
        init_scale = 2 ** 16
        growth_factor = 2
        backoff_factor = 0.5
        scaler = NpuShardedGradScaler(init_scale=init_scale,
                                      backoff_factor=backoff_factor,
                                      growth_factor=growth_factor,
                                      growth_interval=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        for step in range(4):
            loss = model(torch.ones(2, 2).to(device)).sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % 2 == 0:  # Overflow every 2 steps
                self.assertTrue(self.float_close(scaler.get_scale(), init_scale * backoff_factor))
            else:
                self.assertTrue(self.float_close(scaler.get_scale(), init_scale))

    @skipIfUnsupportMultiNPU(2)
    def test_sharded_grad_scaler_cpu(self):
        '''
        Testcase for cpu-offload FSDP sharded grad scaler.
        We init scale to 2, scale will increse every step (as not overflow on cpu),
        then check if the scale is updated correctly.
        '''
        # NpuShardedGradScaler also supports CPU as FSDP maybe offload to cpu.
        self.dist_init()
        device = "cpu"
        model = MyModule().to(device)
        # Set the update interval to 1 to make the scale change more frequently.
        init_scale = 2
        growth_factor = 2
        backoff_factor = 0.5
        scaler = NpuShardedGradScaler(init_scale=init_scale,
                                      backoff_factor=backoff_factor,
                                      growth_factor=growth_factor,
                                      growth_interval=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        for step in range(4):
            loss = model(torch.ones(2, 2).to(device)).sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # CPU finite and will increase the scale every step.
            self.assertTrue(self.float_close(scaler.get_scale(), init_scale * (growth_factor ** (step + 1))))


if __name__ == "__main__":
    run_tests()
