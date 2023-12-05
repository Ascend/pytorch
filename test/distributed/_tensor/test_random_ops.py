import itertools
from unittest import skip

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed._tensor.random as random

from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor._utils import compute_local_shape_and_global_offset
from torch.distributed._tensor.api import distribute_tensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed._tensor.random import is_rng_supported_mesh, manual_seed

from torch.distributed.distributed_c10d import broadcast_object_list

from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase

import torch_npu
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import run_tests


class DistTensorRandomInitTest(DTensorTestBase):
    def _run_init_op(self, init_op, *args, **kwargs):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        input_size = (8, 4)

        # NOTE: currently random initialization on npu device has different
        # behavior from other devices. Unify the test once the behavior is unified.
        if not is_rng_supported_mesh(device_mesh):
            input_tensor = torch.randn(*input_size, device=self.device_type)
            dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
            local_tensor_clone = torch.clone(input_tensor)
            torch.manual_seed(self.rank)
            local_tensor_clone = init_op(local_tensor_clone, *args, **kwargs)
            torch.manual_seed(self.rank)
            dtensor = init_op(dtensor, *args, **kwargs)
            self.assertEqual(local_tensor_clone, dtensor.to_local())
        else:
            # create DTensor from Tensor
            _tensor = torch.empty(*input_size, device="npu")
            dtensor = distribute_tensor(_tensor, device_mesh, [Shard(1)])

            # DTensor random init
            dtensor = init_op(dtensor, *args, **kwargs)
            local_tensor = dtensor.to_local()

            # allgather the local tensors
            dtensor = dtensor.redistribute(device_mesh, [Replicate()])
            local_tensor_gathered = dtensor.to_local()

            # compare with local tensors from other ranks
            for other_rank in range(self.world_size):
                if self.rank != other_rank:
                    slice_idx = [
                        slice(input_size[0]),
                        slice(
                            other_rank * input_size[1], (other_rank + 1) * input_size[1]
                        ),
                    ]
                    # other rank should have a different local tensor
                    self.assertNotEqual(local_tensor_gathered[slice_idx], local_tensor)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_init_ops(self):
        self._run_init_op(
            torch.nn.init.kaiming_uniform_,
            a=0,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        self._run_init_op(torch.nn.init.normal_, mean=1.5, std=0.8)
        self._run_init_op(torch.nn.init.uniform_, a=0, b=1.2)


class DistTensorRandomOpTest(DTensorTestBase):
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_rng_tracker_init(self):
        torch.npu.manual_seed(self.rank)
        object_list = [torch.npu.initial_seed()]
        broadcast_object_list(object_list)
        seed_from_rank_0 = int(object_list[0])

        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # seed synchronization happens after the first `distribute_tensor` call
        dtensor = distribute_tensor(
            torch.empty([self.world_size], device="npu"), device_mesh, [Shard(0)]
        )
        self.assertEqual(seed_from_rank_0, random._rng_tracker.get_seed("parallel-rng"))

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_manual_seed(self):
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        manual_seed(1234, device_mesh)
        self.assertEqual(1234, random._rng_tracker.get_seed("parallel-rng"))
        with self.assertRaisesRegex(RuntimeError, "different seed values"):
            manual_seed(self.rank, device_mesh)

    @skip("OffsetBaseRNGTracker needs to support cuda-like device")
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_deterministic_rand_1d(self):
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        size = [4, 4 * self.world_size]

        dtensor = torch.distributed._tensor.rand(
            size, device_mesh=device_mesh, placements=[Shard(1)]
        )
        local_tensor = funcol.all_gather_tensor(
            dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )

        # compare with local tensors from other ranks
        self_slice = slice(4 * self.rank, 4 * self.rank + 4)
        for other_rank in range(self.world_size):
            if self.rank != other_rank:
                # other rank should have an identical local tensor
                other_slice = slice(4 * other_rank, 4 * other_rank + 4)
                self.assertNotEqual(
                    local_tensor[self_slice, :],
                    local_tensor[other_slice, :],
                )

        torch.manual_seed(self.rank)
        torch.npu.manual_seed(self.rank)
        dtensor = torch.distributed._tensor.rand(
            size, device_mesh=device_mesh, placements=[Replicate()]
        )
        local_tensor = funcol.all_gather_tensor(
            dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )

        # compare with local tensors from other ranks
        self_slice = slice(4 * self.rank, 4 * self.rank + 4)
        for other_rank in range(self.world_size):
            if self.rank != other_rank:
                # other rank should have an identical local tensor
                other_slice = slice(4 * other_rank, 4 * other_rank + 4)
                self.assertEqual(
                    local_tensor[self_slice, :],
                    local_tensor[other_slice, :],
                )

    @skip("OffsetBaseRNGTracker needs to support cuda-like device")
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_deterministic_uniform_2d(self):
        mesh = torch.arange(self.world_size).reshape(2, 2)
        device_mesh = DeviceMesh(self.device_type, mesh)
        dtensor = distribute_tensor(
            torch.empty(
                *[self.world_size for _ in mesh.size()], device=self.device_type
            ),
            device_mesh,
            [Replicate(), Replicate()],
        )

        placements_list = [  # this list of placements should be enough to cover
            [Shard(0), Shard(1)],
            [Shard(1), Shard(0)],
            [Shard(0), Replicate()],
            [Replicate(), Shard(0)],
            [Shard(1), Replicate()],
            [Replicate(), Shard(1)],
            [Replicate(), Replicate()],
        ]

        shard_index_list = [
            {0: 0, 1: 1, 2: 2, 3: 3},
            {0: 0, 1: 2, 2: 1, 3: 3},
            {0: 0, 1: 0, 2: 1, 3: 1},
            {0: 0, 1: 1, 2: 0, 3: 1},
            {0: 0, 1: 0, 2: 1, 3: 1},
            {0: 0, 1: 1, 2: 0, 3: 1},
            {0: 0, 1: 0, 2: 0, 3: 0},
        ]

        coordinate = device_mesh.get_coordinate()
        self.assertTrue(coordinate)

        for placements, shard_index in zip(placements_list, shard_index_list):
            dtensor = dtensor.redistribute(device_mesh, placements)

            # check shard information is correct
            shard_coord = [
                coordinate[mesh_dim] if mesh_dim >= 0 else 0
                for mesh_dim in dtensor._spec.dim_map
            ]

            shard_size = [
                device_mesh.size(mesh_dim) if mesh_dim >= 0 else 1
                for mesh_dim in dtensor._spec.dim_map
            ]

            shard_linear_idx = random._rng_tracker._calc_shard_linear_idx(
                shard_coord, shard_size
            )
            self.assertEqual(shard_linear_idx, shard_index[self.rank])

            # compute local size and offset
            _, local_shard_offset = compute_local_shape_and_global_offset(
                dtensor.shape, device_mesh, placements
            )

            # get the local shard size and local shard offset for each shard
            # local_shard_list_on_dim[i] has the list of all shards on that dim
            # as a tuple (local_shard_offset, local_shard_size)
            dtensor_shape = dtensor.shape
            local_shard_list_on_dim = [[(0, dim)] for dim in dtensor_shape]
            for idx, placement in enumerate(placements):
                if isinstance(placement, Shard):
                    mesh_dim_size = device_mesh.size(idx)
                    shard_dim = placement.dim
                    local_shard_list_on_dim[shard_dim] = []
                    for shard_idx_on_dim in range(mesh_dim_size):
                        shard_size, shard_offset = placement._local_shard_size_on_dim(
                            dtensor_shape[shard_dim],
                            mesh_dim_size,
                            shard_idx_on_dim,
                            return_offset=True,
                        )
                        local_shard_list_on_dim[shard_dim].append(
                            (shard_offset, shard_size)
                        )

            local_shard_comb = itertools.product(*local_shard_list_on_dim)

            # random op call
            dtensor.uniform_(0, 1)

            # the local shard
            local_tensor = dtensor.to_local()
            # allgather the local tensors
            dtensor = dtensor.redistribute(
                device_mesh, [Replicate(), Replicate(), Replicate()]
            )
            local_tensor_gathered = dtensor.to_local()

            # compare local tensor with each other shard
            for other_local_shard in local_shard_comb:
                other_local_shard_offset, _ = zip(*other_local_shard)
                slice_idx = [
                    slice(offset, offset + size) for offset, size in other_local_shard
                ]
                if local_shard_offset == other_local_shard_offset:
                    self.assertEqual(local_tensor_gathered[slice_idx], local_tensor)
                else:
                    self.assertNotEqual(local_tensor_gathered[slice_idx], local_tensor)

    @skip("OffsetBaseRNGTracker needs to support cuda-like device")
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_meta_tensor_init(self):
        # test suite sets each rank's seed to the same value but in actual
        # execution the default random seed will be different (a random value).
        # The DTensor random ops will use the same random seed even though the
        # torch random generator keeps different seeds on ranks. This ensures
        # that Replicate DTensor will have the same initialized results
        # across ranks.
        torch.npu.manual_seed(self.rank)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        size = [1024, 2048]
        meta_dtensor = distribute_tensor(
            torch.empty(*size, device="meta"), device_mesh, [Replicate()]
        )
        self.assertTrue(meta_dtensor.is_meta)
        dtensor = torch.empty_like(meta_dtensor, device=self.device_type)

        # disable the distribute region for RNG
        random._rng_tracker.distribute_region_enabled = False
        dtensor.uniform_()

        # allgather the local tensors
        local_tensor = funcol.all_gather_tensor(
            dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )

        # compare with local tensors from other ranks
        self_slice = slice(1024 * self.rank, 1024 * self.rank + 1024)
        for other_rank in range(self.world_size):
            # the RNG result on each rank differs even they're supposed
            # to be replicated
            if self.rank != other_rank:
                other_slice = slice(1024 * other_rank, 1024 * other_rank + 1024)
                self.assertNotEqual(
                    local_tensor[self_slice, :], local_tensor[other_slice, :]
                )

        # enable the distribute region for RNG
        random._rng_tracker.distribute_region_enabled = True
        self.assertTrue(meta_dtensor.is_meta)
        dtensor = torch.empty_like(meta_dtensor, device=self.device_type)
        dtensor.uniform_()

        # allgather the local tensors
        local_tensor = funcol.all_gather_tensor(
            dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )

        # compare with local tensors from other ranks
        for other_rank in range(self.world_size):
            # the RNG result on each rank are the same because they're replicated
            if self.rank != other_rank:
                # other rank should have an identical local tensor
                other_slice = slice(1024 * other_rank, 1024 * other_rank + 1024)
                self.assertEqual(
                    local_tensor[self_slice, :], local_tensor[other_slice, :]
                )


if __name__ == "__main__":
    run_tests()