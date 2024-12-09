from functools import wraps
import itertools
import os
import types
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.npu.amp.sharded_grad_scaler import _ShardedGradScaler as NpuShardedGradScaler

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp.wrap import _Policy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    LocalOptimStateDictConfig,
    LocalStateDictConfig,
    MixedPrecision,
    OptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictSettings,
    StateDictType,
)


class FSDPTestConfig(object):
    def __init__(self) -> None:
        self.sharding_strategy: Optional[ShardingStrategy] = None
        self.cpu_offload: Optional[CPUOffload] = None
        self.auto_wrap_policy: Optional[Union[Callable, _Policy]] = None
        self.backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE
        self.mixed_precision: Optional[MixedPrecision] = None
        self.ignored_modules: Optional[Iterable[torch.nn.Module]] = None
        self.param_init_fn: Optional[Callable[[nn.Module], None]] = None
        self.device_id: Optional[Union[int, torch.device]] = None
        self.sync_module_states: bool = False
        self.forward_prefetch: bool = False
        self.limit_all_gathers: bool = False
        self.use_orig_params: bool = False
        self.ignored_modules: Optional[Iterable[torch.nn.Module]] = None

    def __repr__(self) -> str:
        msg = "FSDP Config:[\n"
        for k, v in self.__dict__.items():
            msg += f"    {k}: {v}\n"
        msg += "]"
        return msg


def get_wrap_policies() -> List[_Policy]:
    def always_true_policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
        return True

    def always_false_policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
        return False

    return [None, always_true_policy, always_false_policy]


def get_mixed_precision_configs() -> List[MixedPrecision]:
    return [None,
            MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=False)
            ]


def get_cpu_offload_configs() -> List[CPUOffload]:
    return [None, CPUOffload(offload_params=True), CPUOffload(offload_params=False)]


def named_product(**items: Iterable[Any]) -> Generator[Dict[str, Any], None, None]:
    keys = list(items.keys())
    values = list(items.values())
    for values in itertools.product(*values):
        yield dict(zip(keys, values))


def testcase_configs() -> Generator[FSDPTestConfig, None, None]:
    for params in named_product(use_orig_params=[True, False],
                                auto_wrap_policy=get_wrap_policies(),
                                forward_prefetch=[True, False],
                                backward_prefetch=[BackwardPrefetch.BACKWARD_PRE, BackwardPrefetch.BACKWARD_POST],
                                mixed_precision=get_mixed_precision_configs(),
                                sync_module_states=[True, False],
                                cpu_offload=get_cpu_offload_configs(),
                                limit_all_gathers=[True, False]):
        test_config = FSDPTestConfig()
        for k, v in params.items():
            setattr(test_config, k, v)
        yield test_config


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 2)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)


def run_testcase_with_config(device, config: FSDPTestConfig) -> None:
    model = FSDP(MyModule(),
                 device_id=torch.device(device),
                 use_orig_params=config.use_orig_params,
                 cpu_offload=config.cpu_offload,
                 auto_wrap_policy=config.auto_wrap_policy,
                 backward_prefetch=config.backward_prefetch,
                 mixed_precision=config.mixed_precision,
                 sync_module_states=config.sync_module_states,
                 forward_prefetch=config.forward_prefetch,
                 limit_all_gathers=config.limit_all_gathers,
                 ignored_modules=None)
    scaler = NpuShardedGradScaler(growth_interval=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for _ in range(5):
        x = model(torch.ones(2, 2).to(device))
        loss = x.sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    rank = device.split(":")[1]
    torch.save(model.state_dict(), f"test_fsdp_with_many_config{rank}.pt")
    model.load_state_dict(torch.load(f"test_fsdp_with_many_config{rank}.pt"))


class TestFSDP(TestCase):
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

    @skipIfUnsupportMultiNPU(2)
    def test_fsdp_with_many_config(self):
        '''
        Testcase for FSDP with different configs and expect non-raise error.
        '''
        self.dist_init()
        device = f"npu:{self.rank}"
        for config in testcase_configs():
            try:
                if self.rank == 0:
                    print(f"Running FSDP testcase with config: {config}")
                run_testcase_with_config(device, config)
            except Exception as e:
                self.assertTrue(False, f"Error when running with config: {config}, and error is {e}")


if __name__ == "__main__":
    run_tests()
