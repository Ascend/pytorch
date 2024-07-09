from functools import wraps
import io
import os
import types

import torch
import torch_npu


import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)

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
        os.environ['MASTER_PORT'] = '29589'
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

    @skipIfUnsupportMultiNPU(2)
    def test_torch_save_load(self):
        self.dist_init()
        device = torch.device(f"npu:{self.rank}")
        model = FSDP(MyModule(), device_id=device)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = model.state_dict()
            checkpoint = io.BytesIO()
            torch.save(state_dict, checkpoint)
            checkpoint.seek(0)
            state_dict_saved = torch.load(checkpoint)

            for k, v in state_dict_saved.items():
                if isinstance(v, ShardedTensor):
                    self.assertEqual(
                        v._local_shards[0].tensor, state_dict[k]._local_shards[0].tensor)
                else:
                    self.assertEqual(v, state_dict[k])


if __name__ == "__main__":
    run_tests()
