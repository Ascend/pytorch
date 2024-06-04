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
    all_reduce,
    all_to_all_single,
    AsyncCollectiveTensor,
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
        torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    @skipIfUnsupportMultiNPU(2)
    def test_all_reduce_single(self) -> None:
        self._init_process_group()

        input_ = torch.full((10, 10), float(self.rank), device=self.device)
        output = torch.ops._c10d_functional.all_reduce(
            input_,
            "max",
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        assert id(output) != id(input_)
        expect = max(self.ranks)
        assert output.eq(expect).all()

        # Test Python API and AsyncCollectiveTensor
        output = all_reduce(
            input_,
            "max",
            "default",
        )
        assert isinstance(output, AsyncCollectiveTensor)
        assert not output.completed
        assert output.eq(expect).all()
        assert output.completed

    @skipIfUnsupportMultiNPU(2)
    def test_all_reduce_single_(self) -> None:
        self._init_process_group()

        input_ = torch.full((10, 10), float(self.rank), device=self.device)
        output = torch.ops._c10d_functional.all_reduce_(
            input_,
            "max",
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        assert id(output) == id(input_)
        expect = max(self.ranks)
        assert output.eq(expect).all()

    @skipIfUnsupportMultiNPU(2)
    def test_all_to_all_single(self) -> None:
        self._init_process_group()
        torch.npu.set_device(self.device)

        torch.manual_seed(42)
        send_sz_matrix = torch.randint(0, 20, (self.world_size, self.world_size))

        input_split_sizes = send_sz_matrix[self.rank].tolist()
        output_split_sizes = send_sz_matrix[:, self.rank].tolist()
        input_ = torch.full((sum(input_split_sizes),), float(self.rank)).npu()

        output = torch.ops._c10d_functional.all_to_all_single(
            input_,
            output_split_sizes,
            input_split_sizes,
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        expect = torch.cat(
            [
                torch.full((sz,), float(rank)).npu()
                for rank, sz in enumerate(output_split_sizes)
            ]
        )
        assert output.eq(expect).all()

        # Test Python API and AsyncCollectiveTensor
        output = all_to_all_single(
            input_, output_split_sizes, input_split_sizes, "default"
        )
        assert not output.completed
        assert output.eq(expect).all()
        assert output.completed

    @skipIfUnsupportMultiNPU(2)
    def test_broadcast(self) -> None:
        self._init_process_group()

        input_ = torch.full((10, 10), float(self.rank), device=self.device)
        output = torch.ops._c10d_functional.broadcast(
            input_,
            1,
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        assert id(output) != id(input_)
        expect = 1
        assert output.eq(expect).all()

        # Test Python API and AsyncCollectiveTensor
        output = funcol.broadcast(
            input_,
            1,
            "default",
        )
        assert isinstance(output, AsyncCollectiveTensor)
        assert not output.completed
        assert output.eq(expect).all()
        assert output.completed

    @skipIfUnsupportMultiNPU(2)
    def test_unwaited(self) -> None:
        # Verify that the process can terminate gracefully
        # even with unwaited tensors
        self._init_process_group()

        input_ = torch.full((10, 10), float(self.rank), device=self.device)
        output = torch.ops._c10d_functional.all_reduce(
            input_,
            "max",
            "default",
        )

if __name__ == "__main__":
    run_tests()
