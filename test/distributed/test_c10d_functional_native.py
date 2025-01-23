# Owner(s): ["module: c10d"]
from typing import List

import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed._functional_collectives import (
    wait_tensor,
    all_gather_into_tensor_coalesced,
    all_reduce_coalesced,
    reduce_scatter_tensor_coalesced,
)
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
)
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    run_tests,
)
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
import torch_npu


def load_test_module(name):
    testdir = Path(__file__).absolute().parent.parent
    with mock.patch("sys.path", [*sys.path, str(testdir)]):
        return SourceFileLoader(
            name, str(testdir / f"{name.replace('.', '/')}.py")
        ).load_module()


class TestWithHccl(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def ranks(self) -> List[int]:
        return list(range(self.world_size))

    @property
    def device(self) -> torch.device:
        return torch.device(f"npu:{self.rank}")

    def _init_process_group(self) -> None:
        # Allow testing aoti after torch.compile
        torch._inductor.config.triton.store_cubin = True
        torch._inductor.config.debug = True

        torch.npu.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="hccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

    @skipIfUnsupportMultiNPU(2)
    def test_all_reduce_coalesced(self) -> None:
        self._init_process_group()

        inputs = [
            torch.full((i, i), float(self.rank * i), device=self.device)
            for i in range(10)
        ]
        # Test Python API and AsyncCollectiveTensor
        outputs = all_reduce_coalesced(
            inputs,
            "sum",
            dist.distributed_c10d._get_default_group()
        )
        for i, (output, input) in enumerate(zip(outputs, inputs)):
            wait_tensor(output)
            assert output.eq(sum(self.ranks) * i).all()

    @skipIfUnsupportMultiNPU(2)
    def test_all_gather_into_tensor_coalesced(self) -> None:
        self._init_process_group()

        inputs = [
            torch.full((10, 10), float(self.rank * i), device=self.device)
            for i in range(10)
        ]
        expect = [
            torch.cat(
                [
                    torch.full((10, 10), float(rank) * i, device=self.device)
                    for rank in self.ranks
                ]
            )
            for i in range(10)
        ]

        # Test Python API and AsyncCollectiveTensor
        outputs = all_gather_into_tensor_coalesced(
            inputs,
            dist.distributed_c10d._get_default_group()
        )
        for i, output in enumerate(outputs):
            wait_tensor(output)
            assert output.eq(expect[i]).all()

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_tensor_coalesced(self) -> None:
        self._init_process_group()

        inputs = [torch.tensor(self.ranks, device=self.device) * i for i in range(10)]

        # Test Python API and AsyncCollectiveTensor
        outputs = reduce_scatter_tensor_coalesced(
            inputs,
            "sum",
            [0] * 10,
            dist.distributed_c10d._get_default_group(),
        )
        for i, output in enumerate(outputs):
            wait_tensor(output)
            assert output.eq(self.rank * i * self.world_size).all()

if __name__ == "__main__":
    run_tests()
