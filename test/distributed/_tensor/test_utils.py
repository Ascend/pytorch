import itertools
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import (
    local_tensor_mode,
    LocalTensor,
    LocalTensorMode,
)
from torch.distributed._tensor import distribute_tensor
from torch.distributed.tensor._utils import (
    compute_local_shape_and_global_offset,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor._utils import ExplicitRedistributionContext
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import Replicate, Shard

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase

import torch_npu
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU


class UtilTest(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_compute_local_shape_and_global_offset_1D(self):
        one_d_placements = [[Shard(0)], [Replicate()]]

        for placements in one_d_placements:
            mesh_tensor = torch.arange(self.world_size)
            device_mesh = DeviceMesh(self.device_type, mesh_tensor)
            global_tensor = torch.arange(64).view(8, 8)
            global_shape = global_tensor.size()

            dtensor = distribute_tensor(global_tensor, device_mesh, placements)
            local_size, global_offset = compute_local_shape_and_global_offset(
                global_shape, device_mesh, placements
            )

            dim0_start = global_offset[0]
            dim0_end = global_offset[0] + local_size[0]

            # Check the local tensor of dtensor is exactly the same
            # if we slice the global_tensor with local_size and global_offset
            self.assertEqual(
                dtensor.to_local(),
                global_tensor[dim0_start:dim0_end],
            )

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_compute_local_shape_and_global_offset_2D(self):
        two_d_placements_options = [Shard(0), Shard(1), Replicate()]
        # Generating 6 two-d placements combinations
        two_d_placements = list(
            itertools.combinations_with_replacement(two_d_placements_options, 2)
        )

        for placements in two_d_placements:
            # mesh: 2 * 4
            mesh_tensor = torch.arange(self.world_size).reshape(2, 4)
            device_mesh = DeviceMesh(self.device_type, mesh_tensor)
            global_tensor = torch.arange(64).view(8, 8)
            global_shape = global_tensor.size()

            dtensor = distribute_tensor(global_tensor, device_mesh, placements)
            local_size, global_offset = compute_local_shape_and_global_offset(
                global_shape, device_mesh, placements
            )

            dim0_start = global_offset[0]
            dim0_end = global_offset[0] + local_size[0]
            dim1_start = global_offset[1]
            dim1_end = global_offset[1] + local_size[1]

            # Check the local tensor of dtensor is exactly the same
            # if we slice the global_tensor with local_size and global_offset
            self.assertEqual(
                dtensor.to_local(),
                global_tensor[dim0_start:dim0_end, dim1_start:dim1_end],
            )


class LocalTensorTestBase(TestCase):
    def assertEqual(self, lhs, rhs, **kwargs):
        mode = local_tensor_mode()
        with nullcontext() if mode is None else mode.disable():
            if isinstance(lhs, LocalTensor) and isinstance(rhs, LocalTensor):
                assert isinstance(lhs, LocalTensor) and isinstance(rhs, LocalTensor)
                super().assertEqual(lhs._ranks, rhs._ranks)
                for r in lhs._ranks:
                    super().assertEqual(
                        lhs._local_tensors[r],
                        rhs._local_tensors[r],
                        lambda m: f"rank {r}: {m}",
                    )
            elif isinstance(lhs, LocalTensor) or isinstance(rhs, LocalTensor):
                lhs, rhs = (lhs, rhs) if isinstance(lhs, LocalTensor) else (rhs, lhs)
                for r in lhs._ranks:
                    super().assertEqual(
                        lhs._local_tensors[r], rhs, lambda m: f"rank {r}: {m}"
                    )
            else:
                return super().assertEqual(lhs, rhs, **kwargs)

    @property
    def world_size(self):
        raise NotImplementedError("override world-size in your subclass")

    def build_device_mesh(self) -> DeviceMesh:
        return init_device_mesh("cpu", (self.world_size,))

    def setUp(self):
        super().setUp()
        torch.distributed.init_process_group(
            # TODO: test other ranks too
            "fake",
            rank=0,
            world_size=self.world_size,
        )

    def tearDown(self):
        super().tearDown()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass


class TestExplicitRedistribute(LocalTensorTestBase):
    @property
    def world_size(self):
        return 4

    def test_explicit_matmul(self):
        with LocalTensorMode(self.world_size):
            device_mesh = self.build_device_mesh()
            dim = 128
            x = torch.randn(8, dim, requires_grad=True)
            A = torch.randn(dim, dim, requires_grad=True)

            # Prepare DTensors
            dx = distribute_tensor(x, device_mesh, [Shard(0)])
            dA = distribute_tensor(A, device_mesh, [Shard(0)])

            # implicit redistribute works as usual by default
            with CommDebugMode() as comm_mode:
                torch.matmul(dx, dA)
            self.assertEqual(comm_mode.get_total_counts(), 1)

            # explicit redistribute works too
            with ExplicitRedistributionContext():
                with self.assertRaisesRegex(RuntimeError, "Implicit redistribution"):
                    torch.matmul(dx, dA)
            with ExplicitRedistributionContext(mode="warn"):
                with self.assertLogs(
                    torch.distributed.tensor._utils.logger, level="WARN"
                ) as captured:
                    torch.matmul(dx, dA)
                    self.assertEqual(len(captured.output), 1)
                    self.assertRegex(
                        captured.output[0],
                        r"WARNING:.*Implicit redistribution occurred",
                    )
                    # TODO enable this once fixing the issue that op_info.schema is None in some calls to
                    # redistribute_local_tensor
                    # self.assertRegex(
                    #     captured.output[0],
                    #     r".*aten\.mm\.default.*",
                    # )

            # explicit redistribute allows manual redistribute
            with ExplicitRedistributionContext():
                dA_repl = dA.redistribute(device_mesh, [Replicate()])
                torch.matmul(dx, dA_repl)

            dx = distribute_tensor(x, device_mesh, [Shard(0)])
            dA = distribute_tensor(A, device_mesh, [Replicate()])
            with ExplicitRedistributionContext(strict=True):
                dY = torch.matmul(dx, dA_repl)
                loss = dY.sum()

                # we now see the error during backwards
                with self.assertRaisesRegex(RuntimeError, "Implicit redistribution"):
                    loss.backward(retain_graph=True)

                with ExplicitRedistributionContext(strict=False):
                    # but since it's a 'free' redistribute, we can still do it under non-strict mode
                    loss.backward(retain_graph=True)

                with ExplicitRedistributionContext(enable=False):
                    # and we can disable
                    loss.backward(retain_graph=True)

                # and re-enable
                with self.assertRaisesRegex(RuntimeError, "Implicit redistribution"):
                    loss.backward(retain_graph=True)


if __name__ == "__main__":
    run_tests()
