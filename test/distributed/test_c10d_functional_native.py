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
    all_gather_into_tensor_coalesced,
    all_gather_tensor,
    all_reduce,
    all_reduce_coalesced,
    all_to_all_single,
    AsyncCollectiveTensor,
    reduce_scatter_tensor,
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

    @skipIfUnsupportMultiNPU(2)
    def test_all_reduce_coalesced(self) -> None:
        self._init_process_group()

        inputs = [
            torch.full((i, i), float(self.rank * i), device=self.device)
            for i in range(10)
        ]
        outputs = torch.ops._c10d_functional.all_reduce_coalesced(
            inputs,
            "sum",
            "default",
        )
        for i, (output, input) in enumerate(zip(outputs, inputs)):
            output = torch.ops._c10d_functional.wait_tensor(output)
            assert id(output) != id(input)
            assert output.eq(sum(self.ranks) * i).all()

        # Test Python API and AsyncCollectiveTensor
        outputs = all_reduce_coalesced(
            inputs,
            "sum",
            "default",
        )
        for i, (output, input) in enumerate(zip(outputs, inputs)):
            assert not output.completed
            assert output.eq(sum(self.ranks) * i).all()
            assert output.completed

    @skipIfUnsupportMultiNPU(2)
    def test_all_reduce_coalesced_(self) -> None:
        self._init_process_group()

        inputs = [
            torch.full((i, i), float(self.rank * i), device=self.device)
            for i in range(10)
        ]
        outputs = torch.ops._c10d_functional.all_reduce_coalesced_(
            inputs,
            "sum",
            "default",
        )
        for i, (output, input) in enumerate(zip(outputs, inputs)):
            output = torch.ops._c10d_functional.wait_tensor(output)
            assert id(output) == id(input)
            assert output.eq(sum(self.ranks) * i).all()

    @skipIfUnsupportMultiNPU(2)
    def test_all_gather_into_tensor_single(self) -> None:
        self._init_process_group()

        input = torch.full((10, 10), float(self.rank), device=self.device)
        output = torch.ops._c10d_functional.all_gather_into_tensor(
            input,
            self.world_size,
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        expect = torch.cat(
            [
                torch.full((10, 10), float(rank), device=self.device)
                for rank in self.ranks
            ]
        )
        assert torch.allclose(output, expect)
        assert output.eq(expect).all()

        # Test out-variant of all_gather_into_tensor
        output = torch.empty(expect.shape, device=self.device)
        output = torch.ops._c10d_functional.all_gather_into_tensor_out(
            input,
            self.world_size,
            "default",
            out=output,
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        assert torch.allclose(output, expect)
        assert output.eq(expect).all()

        # Test Python API and AsyncCollectiveTensor
        output = all_gather_tensor(
            input,
            0,
            "default",
        )
        assert isinstance(output, AsyncCollectiveTensor)
        assert not output.completed
        assert output.eq(expect).all()
        assert output.completed

    @skipIfUnsupportMultiNPU(2)
    def test_all_gather_into_tensor_coalesced(self) -> None:
        self._init_process_group()

        inputs = [
            torch.full((10, 10), float(self.rank * i), device=self.device)
            for i in range(10)
        ]
        outputs = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
            inputs,
            self.world_size,
            "default",
        )
        expect = [
            torch.cat(
                [
                    torch.full((10, 10), float(rank) * i, device=self.device)
                    for rank in self.ranks
                ]
            )
            for i in range(10)
        ]
        for i, output in enumerate(outputs):
            output = torch.ops._c10d_functional.wait_tensor(output)
            assert output.eq(expect[i]).all()

        # Test Python API and AsyncCollectiveTensor
        outputs = all_gather_into_tensor_coalesced(
            inputs,
            "default",
        )
        for i, output in enumerate(outputs):
            assert not output.completed
            assert output.eq(expect[i]).all()
            assert output.completed

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_tensor_single(self) -> None:
        self._init_process_group()

        input = torch.tensor(self.ranks, device=self.device)
        output = torch.ops._c10d_functional.reduce_scatter_tensor(
            input,
            "sum",
            self.world_size,
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        assert output.eq(self.rank * self.world_size).all()

        # Test Python API and AsyncCollectiveTensor
        output = reduce_scatter_tensor(
            input,
            "sum",
            0,
            "default",
        )
        assert isinstance(output, AsyncCollectiveTensor)
        assert not output.completed
        assert output.eq(self.rank * self.world_size).all()
        assert output.completed

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_tensor_coalesced(self) -> None:
        self._init_process_group()

        inputs = [torch.tensor(self.ranks, device=self.device) * i for i in range(10)]
        outputs = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
            inputs,
            "sum",
            self.world_size,
            "default",
        )
        for i, output in enumerate(outputs):
            output = torch.ops._c10d_functional.wait_tensor(output)
            assert output.eq(self.rank * i * self.world_size).all()

        # Test Python API and AsyncCollectiveTensor
        outputs = reduce_scatter_tensor_coalesced(
            inputs,
            "sum",
            [0] * 10,
            "default",
        )
        for i, output in enumerate(outputs):
            assert not output.completed
            assert output.eq(self.rank * i * self.world_size).all()
            assert output.completed

if __name__ == "__main__":
    run_tests()
