# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import os

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh, init_device_mesh
from torch.distributed.distributed_c10d import (
    _get_default_group,
    _world,
    get_global_rank,
    get_world_size,
    init_process_group,
    is_initialized,
    ProcessGroup,
)
from torch.distributed.tensor._collective_utils import (
    mesh_broadcast,
    mesh_scatter,
    unpad_tensor,
)
from torch.distributed.tensor.placement_types import _Partial, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase
from torch.testing._internal.distributed.fake_pg import FakeStore

import torch_npu
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import run_tests


def _get_device_type(world_size):
    if (
        torch.npu.is_available()
        and torch.npu.device_count() >= world_size
        and torch.distributed.is_hccl_available()
    ):
        device_type = "npu"
    else:
        device_type = "cpu"
    return device_type


def _set_env_var(addr="localhost", port="29500", world_size=1, rank=0):
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["RANK"] = f"{rank}"


class DeviceMeshTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @skipIfUnsupportMultiNPU(2)
    def test_init_process_group(self):
        device_type = _get_device_type(self.world_size)
        mesh_tensor = torch.arange(2).reshape(2, 1)
        self.assertTrue(not is_initialized())
        _set_env_var(world_size=self.world_size, rank=self.rank)
        DeviceMesh(device_type, mesh_tensor)
        self.assertTrue(is_initialized())
        self.destroy_pg()

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_2d_mesh_non_eager_init_subgroup(self):
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(self.device_type, mesh_shape)

        self.assertEqual(mesh_2d.get_group(0).bound_device_id, None)
        self.assertEqual(mesh_2d.get_group(1).bound_device_id, None)

    # need to refactor the other tests in this file to test both
    # eager_init=True and eager_init=False scenarios.
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_2d_mesh_eager_init_subgroup(self):
        # Test with eager_init=True
        with self.subTest(eager_init=True):
            mesh_shape = (2, self.world_size // 2)
            mesh_2d = init_device_mesh(self.device_type, mesh_shape)

            # when eager init is used, the subgroup is created from hccl comm split and
            # there would be bound_device_id immediately assigned for the subgroup.
            if self.backend == "hccl":
                curr_device = torch.npu.current_device()
                self.assertEqual(mesh_2d.get_group(0).bound_device_id.index, curr_device)
                self.assertEqual(mesh_2d.get_group(1).bound_device_id.index, curr_device)

        # Test with eager_init=False
        with self.subTest(eager_init=False):
            mesh_shape = (2, self.world_size // 2)
            mesh_2d = init_device_mesh(self.device_type, mesh_shape)

            # when eager init is used, the subgroup is created from hccl comm split and
            # there would be bound_device_id immediately assigned for the subgroup.
            if self.backend == "hccl":
                curr_device = torch.npu.current_device()
                self.assertEqual(mesh_2d.get_group(0).bound_device_id.index, curr_device)
                self.assertEqual(mesh_2d.get_group(1).bound_device_id.index, curr_device)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_get_group_and_get_all_groups(self):
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("dp", "tp")
        )

        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]

        self.assertEqual(mesh_2d.get_group(0), mesh_2d.get_group("dp"))
        self.assertEqual(mesh_2d.get_group(1), mesh_2d.get_group("tp"))

        self.assertEqual(mesh_2d.get_group("dp"), dp_mesh.get_group())
        self.assertEqual(mesh_2d.get_group("tp"), tp_mesh.get_group())

        groups = mesh_2d.get_all_groups()
        self.assertEqual(len(groups), 2)
        self.assertTrue(tp_mesh.get_group() in groups)
        self.assertTrue(dp_mesh.get_group() in groups)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_get_local_rank_raises_exception(self):
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("dp", "tp")
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
        ):
            local_rank = mesh_2d.get_local_rank()

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_device_mesh_init_backend(self):
        mesh = DeviceMesh(self.device_type, [1], _init_backend=False)

        with self.assertRaisesRegex(RuntimeError, "process groups not initialized!"):
            mesh.get_group()

        # coordinates should always been populated when init_backend is False, as whenever
        # we call init_backend we should make sure the default pg already created
        mesh.get_coordinate()

    @skipIfUnsupportMultiNPU(2)
    def test_fake_pg_device_mesh(self):
        fake_store = FakeStore()
        init_process_group("fake", store=fake_store, rank=0, world_size=self.world_size)
        device_type = "npu" if torch.npu.is_available() else "cpu"
        mesh = DeviceMesh(device_type, torch.arange(self.world_size))

        local_tensor = torch.randn(2, 8)
        global_tensor = funcol.all_gather_tensor(
            local_tensor, gather_dim=0, group=(mesh, 0)
        )
        self.assertEqual(global_tensor.shape, (self.world_size * 2, 8))

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_from_group_with_global_pg(self):
        # Simple test: check `from_group` from a mesh pg vs. directly
        # initializing via `init_device_mesh`
        ref_global_mesh = init_device_mesh(self.device_type, (self.world_size,))
        mesh_pg = ref_global_mesh.get_group()
        global_mesh = DeviceMesh.from_group(mesh_pg, self.device_type)
        self.assertEqual(ref_global_mesh, global_mesh)
        self.assertEqual(ref_global_mesh._dim_group_infos, global_mesh._dim_group_infos)
        self.assertEqual(
            ref_global_mesh._coordinate_on_dim, global_mesh._coordinate_on_dim
        )
        # Check when `mesh` is passed as well
        global_mesh = DeviceMesh.from_group(
            mesh_pg, self.device_type, mesh=torch.arange(self.world_size)
        )
        self.assertEqual(ref_global_mesh, global_mesh)
        self.assertEqual(ref_global_mesh._dim_group_infos, global_mesh._dim_group_infos)
        self.assertEqual(
            ref_global_mesh._coordinate_on_dim, global_mesh._coordinate_on_dim
        )

    @skipIfUnsupportMultiNPU(2)
    def test_raises_invalid_device_type(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Device type with index is not supported",
        ):
            # test init_device_mesh with an invalid device type that contains a NPU index
            mesh_shape = (2, self.world_size // 2)
            mesh_2d = init_device_mesh(
                "npu:0", mesh_shape=mesh_shape, mesh_dim_names=("dp", "tp")
            )

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_set_mesh_dim_group_options(self):
        device_type = "npu" if torch.npu.is_available() else "cpu"
        _mesh_resources._set_mesh_dim_group_options(1, "fake", None)

        mesh_tensor = torch.arange(2).reshape(2, 1)
        mesh = DeviceMesh(device_type, mesh_tensor)
        # Fake pg only have BackendType as BackendType::CUSTOM.
        self.assertEqual(mesh.get_group(1)._get_backend_name(), "custom")


#DeviceMeshTest with resetting world_size to 4.
class DeviceMeshTestF(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_assert_invalid_mesh_tensor(self):
        mesh = torch.arange(self.world_size).to(self.rank)
        with self.assertRaises(ValueError):
            device_mesh = DeviceMesh(self.device_type, mesh)
    
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_get_local_rank(self):
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("dp", "tp")
        )
        self.assertEqual(mesh_2d.get_local_rank("dp"), mesh_2d.get_local_rank(0))
        self.assertEqual(mesh_2d.get_local_rank("tp"), mesh_2d.get_local_rank(1))

        dp_mesh = mesh_2d["dp"]
        tp_mesh = mesh_2d["tp"]
        self.assertEqual(dp_mesh.get_local_rank(), mesh_2d.get_local_rank("dp"))
        self.assertEqual(tp_mesh.get_local_rank(), mesh_2d.get_local_rank("tp"))

        # Verify flattened mesh local rank correctness.
        flattened_mesh = mesh_2d["dp", "tp"]._flatten()
        self.assertEqual(flattened_mesh.get_local_rank(), self.rank)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_device_mesh_2d(self):
        mesh_tensor = torch.arange(4).reshape(2, 2)
        # construct a npu device mesh
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_all_groups()

        expected_ranks_by_dim = [[[0, 2], [1, 3]], [[0, 1], [2, 3]]]
        for dim, dim_group in enumerate(dim_to_subgroups):
            self.assertTrue(dim < 2)
            dim_ranks = expected_ranks_by_dim[dim]

            dim_group_size = get_world_size(dim_group)
            self.assertIsInstance(dim_group, ProcessGroup)
            self.assertEqual(dim_group_size, 2)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            current_rank_expected_group_ranks = (
                dim_ranks[0] if self.rank in dim_ranks[0] else dim_ranks[1]
            )
            self.assertEqual(global_ranks, current_rank_expected_group_ranks)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_from_group_with_invalid_mesh(self):
        global_pg = _get_default_group()
        global_pg_size = global_pg.size()
        assert global_pg_size == 4, "Test assumes global world size of 4"
        invalid_mesh = [[0, 1], [2, 3]]  # 2D mesh when we need 1D
        regex = r"Invalid mesh \[\[0, 1\], \[2, 3\]\] for ProcessGroup with ranks \[0, 1, 2, 3\]"
        with self.assertRaisesRegex(ValueError, regex):
            DeviceMesh.from_group(global_pg, "npu", invalid_mesh)

        device_mesh = init_device_mesh(self.device_type, (2, 2))
        groups = device_mesh.get_all_groups()
        invalid_mesh = (0, 1, 2, 3)  # 1D mesh when we need 2D
        regex = r"Expects mesh with ndim equal to number of ProcessGroups but got mesh \[0, 1, 2, 3\] and 2 ProcessGroups"
        with self.assertRaisesRegex(ValueError, regex):
            DeviceMesh.from_group(groups, self.device_type, invalid_mesh)


class DeviceMeshTestNDim(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_device_mesh_parent_child_hash(self):
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("DP", "TP")
        )

        mesh_group_1 = torch.arange(0, self.world_size // 2)
        mesh_group_2 = torch.arange(self.world_size // 2, self.world_size)
        ep_mesh_1 = DeviceMesh(self.device_type, mesh_group_1)
        ep_mesh_2 = DeviceMesh(self.device_type, mesh_group_2)
        ep_mesh = ep_mesh_1 if self.rank < self.world_size // 2 else ep_mesh_2
        # ep_mesh is considered different from mesh_2d["TP"]
        self.assertEqual(mesh_2d["TP"]._flatten_mesh_list, ep_mesh._flatten_mesh_list)
        self.assertEqual(mesh_2d["TP"].mesh.shape, ep_mesh.mesh.shape)
        self.assertEqual(mesh_2d["TP"].device_type, ep_mesh.device_type)
        self.assertNotEqual(mesh_2d["TP"].mesh_dim_names, ep_mesh.mesh_dim_names)
        self.assertEqual(mesh_2d["TP"]._thread_id, ep_mesh._thread_id)
        self.assertNotEqual(hash(mesh_2d["TP"]), hash(ep_mesh))
        self.assertNotEqual(mesh_2d["TP"], ep_mesh)

        another_mesh_1 = DeviceMesh(self.device_type, mesh_group_1)
        another_mesh_2 = DeviceMesh(self.device_type, mesh_group_2)
        another_mesh = (
            another_mesh_1 if self.rank < self.world_size // 2 else another_mesh_2
        )
        # another_mesh is considered the same as ep_mesh
        self.assertEqual(ep_mesh._flatten_mesh_list, another_mesh._flatten_mesh_list)
        self.assertEqual(ep_mesh.mesh.shape, another_mesh.mesh.shape)
        self.assertEqual(ep_mesh.device_type, another_mesh.device_type)
        self.assertEqual(ep_mesh.mesh_dim_names, another_mesh.mesh_dim_names)
        self.assertEqual(ep_mesh._thread_id, another_mesh._thread_id)
        self.assertEqual(hash(ep_mesh), hash(another_mesh))
        self.assertEqual(ep_mesh, another_mesh)


#DeviceMeshTestNDim with resetting world_size to 8.
class DeviceMeshTestNDimE(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @skipIfUnsupportMultiNPU(8)
    @with_comms
    def test_device_mesh_nd(self):
        # construct a npu device mesh
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_all_groups()

        for dim, dim_group in enumerate(dim_to_subgroups):
            self.assertTrue(dim < mesh_tensor.ndim)
            dim_ranks = mesh_tensor.swapdims(-1, dim).reshape(-1, 2)

            dim_group_size = get_world_size(dim_group)
            self.assertIsInstance(dim_group, ProcessGroup)
            self.assertEqual(dim_group_size, 2)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            for ranks in dim_ranks:
                if self.rank in ranks:
                    self.assertEqual(global_ranks, ranks.tolist())

    @skipIfUnsupportMultiNPU(8)
    @with_comms
    def test_device_mesh_hash(self):
        mesh_tensor_2d = torch.arange(8).reshape(4, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor_2d)
        mesh2 = DeviceMesh(self.device_type, mesh_tensor_2d)
        self.assertEqual(hash(mesh), hash(mesh2))
        mesh_tensor_3d = torch.arange(8).reshape(2, 2, 2)
        mesh3 = DeviceMesh(self.device_type, mesh_tensor_3d)
        self.assertNotEqual(hash(mesh), hash(mesh3))
        self.assertNotEqual(hash(mesh2), hash(mesh3))

    @skipIfUnsupportMultiNPU(8)
    @with_comms
    def test_get_local_rank_3d(self):
        """
        If we have a 3D mesh and we want to apply dp, pp, tp to it,
        mesh_dim_names = ["dp", "pp", "tp"], and the mesh tensor would be:
        mesh_3d_tensor = [
            [
                [0, 1],
                [2, 3],
            ],
            [
                [4, 5],
                [6, 7],
            ]

        ]
        """
        mesh_shape = (2, 2, 2)
        mesh_3d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("dp", "pp", "tp")
        )

        # tp_rank_0: [0, 2, 4, 6], tp_rank_1: [1, 3, 5, 7]
        tp_rank = mesh_3d.get_local_rank("tp")
        expected_tp_rank = self.rank % 2
        self.assertEqual(tp_rank, expected_tp_rank)

        # pp_rank_0: [0, 1, 4, 5], pp_rank_1: [2, 3, 6, 7]
        pp_rank = mesh_3d.get_local_rank("pp")
        expected_pp_rank = 0 if self.rank % 4 <= 1 else 1
        self.assertEqual(pp_rank, expected_pp_rank)

        # dp_rank_0: [0, 1, 2, 3], dp_rank_1: [4, 5, 6, 7]
        dp_rank = mesh_3d.get_local_rank("dp")
        expected_dp_rank = self.rank // 4
        self.assertEqual(dp_rank, expected_dp_rank)

    @skipIfUnsupportMultiNPU(8)
    @with_comms
    def test_from_group_with_mesh_shape(self):
        """Tests ``from_group`` when passing ``mesh_shape`` as 2D."""
        # Consider two different logical views of the same mesh:
        # - (4, 2) ("dp", "tp") mesh
        # - (2, 2, 2) ("dp_replicate", "dp_shard", "tp") mesh
        mesh_shape = (2, 2, 2)
        mesh_dim_names = ("dp_replicate", "dp_shard", "tp")
        ref_mesh = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        dp_shard_group = ref_mesh["dp_shard"].get_group()
        dp_replicate_group = ref_mesh["dp_replicate"].get_group()

        dp_mesh = DeviceMesh.from_group(
            [dp_replicate_group, dp_shard_group],
            self.device_type,
            mesh=ref_mesh.mesh[:, :, ref_mesh.get_local_rank(2)],
            mesh_dim_names=mesh_dim_names[:2],
        )

        ref_mesh_dp_dim_group_infos = ref_mesh._dim_group_infos[:2]
        for (_, ref_ranks, _), (_, ranks, _) in zip(
            ref_mesh_dp_dim_group_infos, dp_mesh._dim_group_infos
        ):
            self.assertEqual(ref_ranks, ranks)
        # Cannot check directly for mesh equality since parent meshes are not
        # the same since the ref's parent mesh is 3D
        self.assertEqual(dp_mesh["dp_replicate"].mesh, ref_mesh["dp_replicate"].mesh)
        for (_, ref_ranks, _), (_, ranks, _) in zip(
            dp_mesh["dp_replicate"]._dim_group_infos,
            ref_mesh["dp_replicate"]._dim_group_infos,
        ):
            self.assertEqual(ref_ranks, ranks)
        self.assertEqual(dp_mesh["dp_shard"].mesh, ref_mesh["dp_shard"].mesh)
        for (_, ref_ranks, _), (_, ranks, _) in zip(
            dp_mesh["dp_shard"]._dim_group_infos, ref_mesh["dp_shard"]._dim_group_infos
        ):
            self.assertEqual(ref_ranks, ranks)


class InitDeviceMeshTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_init_device_mesh(self):
        mesh_shape = (2, 1)
        mesh_dim_names = ("DP", "TP")
        ref_mesh = DeviceMesh(
            self.device_type,
            torch.arange(2).view(mesh_shape),
            mesh_dim_names=mesh_dim_names,
        )

        # test init_device_mesh with mesh_dim_names
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )
        self.assertEqual(mesh_2d, ref_mesh)
        self.assertEqual(mesh_2d.mesh_dim_names, mesh_dim_names)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_raises_duplicate_mesh_dim_names(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Each mesh_dim_name must be unique.",
        ):
            mesh = init_device_mesh(
                self.device_type,
                (1, 2),
                mesh_dim_names=["dp", "dp"],
            )

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_raises_mesh_shape_mesh_dim_names_mismatch(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "mesh_shape and mesh_dim_names should have same length!",
        ):
            mesh = init_device_mesh(
                self.device_type,
                (2,),
                mesh_dim_names=["dp", "tp"],
            )


class TestDeviceMeshGetItem(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_raises_no_mesh_dim_found(self):
        with self.assertRaisesRegex(
            RuntimeError, "Cannot slice a DeviceMesh without mesh_dim_names!"
        ):
            mesh = init_device_mesh(self.device_type, (1, 2))
            child_mesh = mesh["DP"]

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_raises_invalid_mesh_dim_name(self):
        child_mesh_dim_name = ("PP",)
        with self.assertRaisesRegex(KeyError, "Invalid mesh_dim_name"):
            mesh_dim_names = ("DP", "TP")
            mesh = init_device_mesh(
                self.device_type, (1, 2), mesh_dim_names=mesh_dim_names
            )
            child_mesh = mesh[child_mesh_dim_name]

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_get_item_1d(self):
        mesh = init_device_mesh(self.device_type, (2,), mesh_dim_names=("dp",))
        # Make sure slicing out 1D mesh from a 1D mesh works.
        dp_mesh = mesh["dp"]
        self.assertEqual(dp_mesh, mesh)

        with self.assertRaisesRegex(KeyError, "Invalid mesh_dim_name"):
            dp_mesh = mesh["dim0"]

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_cache_and_reuse_submesh_slice_result(self):
        mesh = init_device_mesh(self.device_type, (1, 2), mesh_dim_names=("dp", "tp"))

        dp_mesh = mesh["dp"]
        ref_pg_count = _world.group_count

        # When we access the "dp" slice again it should not create any new pg.
        # As we are just using the cached result so the pg count should be the same.
        dp_mesh_2 = mesh["dp"]
        self.assertEqual(ref_pg_count, _world.group_count)

        # When we access the "tp" slice, it should not create a new pg, as the "tp" slice would
        # just reuse the parent mesh pg.
        tp_mesh = mesh["tp"]
        self.assertEqual(_world.group_count, ref_pg_count)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_flatten_mesh_3d(self):
        mesh_shape = (1, 1, 2)
        mesh_dim_names = ("dp", "cp", "tp")
        mesh_3d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        # Test flatten contiguous dims
        dp_cp_mesh = mesh_3d["dp", "cp"]
        flattened_dp_cp_mesh = dp_cp_mesh._flatten()
        self.assertEqual(dp_cp_mesh.mesh.flatten(), flattened_dp_cp_mesh.mesh)
        self.assertEqual(flattened_dp_cp_mesh.mesh_dim_names[0], "dp_cp")
        root_mesh = _mesh_resources.get_root_mesh(dp_cp_mesh)
        self.assertEqual(root_mesh, mesh_3d)
        flatten_mesh_root_dims = _mesh_resources.flatten_name_to_root_dims[root_mesh][
            "dp_cp"
        ]
        self.assertEqual(flatten_mesh_root_dims, (0, 1))

        ref_pg_count = _world.group_count
        # Calling flatten again should not create a new pg.
        flattened_dp_cp_mesh_2 = dp_cp_mesh._flatten()
        self.assertEqual(flattened_dp_cp_mesh, flattened_dp_cp_mesh_2)
        self.assertEqual(ref_pg_count, _world.group_count)

        # Test flatten non-contiguous dims
        dp_tp_mesh = mesh_3d["dp", "tp"]
        flattened_dp_tp_mesh = dp_tp_mesh._flatten()
        self.assertEqual(dp_tp_mesh.mesh.flatten(), flattened_dp_tp_mesh.mesh)
        self.assertEqual(flattened_dp_tp_mesh.mesh_dim_names[0], "dp_tp")
        root_mesh = _mesh_resources.get_root_mesh(dp_tp_mesh)
        self.assertEqual(root_mesh, mesh_3d)
        flatten_mesh_root_dims = _mesh_resources.flatten_name_to_root_dims[root_mesh][
            "dp_tp"
        ]
        self.assertEqual(flatten_mesh_root_dims, (0, 2))

        # Test flatten with a flattened mesh_dim_name
        cp_tp_mesh = mesh_3d["cp", "tp"]
        cp_tp_mesh._flatten("dummy")
        self.assertEqual(mesh_3d["dummy"].mesh_dim_names[0], "dummy")

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_flatten_mesh_4d(self):
        with self.subTest(eager_init=True):
            mesh_shape = (2, 1, 1, 1)
            mesh_dim_names = ("dp_replicate", "dp_shard", "cp", "tp")
            mesh_4d = init_device_mesh(
                self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
            )

            # flatten HSDP and CP into one mesh
            dp_cp_mesh = mesh_4d[mesh_dim_names[:3]]._flatten("dp_cp")
            # check flattened mesh integrity
            self.assertEqual(mesh_4d["dp_cp"].mesh.flatten(), dp_cp_mesh.mesh)
            # check flattened mesh dim names is correct
            self.assertEqual(dp_cp_mesh.mesh_dim_names, ("dp_cp",))
            # check flattened mesh dependency
            self.assertEqual(_mesh_resources.get_root_mesh(dp_cp_mesh), mesh_4d)

        with self.subTest(eager_init=False):
            mesh_shape = (2, 1, 1, 1)
            mesh_dim_names = ("dp_replicate", "dp_shard", "cp", "tp")
            mesh_4d = init_device_mesh(
                self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
            )

            # flatten HSDP and CP into one mesh
            dp_cp_mesh = mesh_4d[mesh_dim_names[:3]]._flatten("dp_cp")
            # check flattened mesh integrity
            self.assertEqual(mesh_4d["dp_cp"].mesh.flatten(), dp_cp_mesh.mesh)
            # check flattened mesh dim names is correct
            self.assertEqual(dp_cp_mesh.mesh_dim_names, ("dp_cp",))
            # check flattened mesh dependency
            self.assertEqual(_mesh_resources.get_root_mesh(dp_cp_mesh), mesh_4d)


#TestDeviceMeshGetItem with resetting world_size to 8.
class TestDeviceMeshGetItemE(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @skipIfUnsupportMultiNPU(8)
    @with_comms
    def test_get_item_2d(self):
        mesh_shape = (2, 4)
        mesh_dim_names = ("DP", "TP")
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        pg_ranks_by_dim_name = {}
        for mesh_dim_name in mesh_dim_names:
            mesh_dim = mesh_dim_names.index(mesh_dim_name)
            pg_ranks_by_dim_name[mesh_dim_name] = mesh_2d.mesh.swapdims(
                -1, mesh_dim
            ).reshape(-1, mesh_2d.mesh.size(mesh_dim))

        tp_mesh = mesh_2d["TP"]
        tp_group_idx = self.rank // 4
        self.assertEqual(tp_mesh.mesh, pg_ranks_by_dim_name["TP"][tp_group_idx])

        dp_mesh = mesh_2d["DP"]
        dp_group_idx = self.rank % 4
        self.assertEqual(mesh_2d["DP"].mesh, pg_ranks_by_dim_name["DP"][dp_group_idx])

    @skipIfUnsupportMultiNPU(8)
    @with_comms
    def test_get_item_3d(self):
        mesh_shape = (2, 2, 2)
        mesh_dim_names = ("Replicate", "Shard", "TP")
        mesh_3d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        tp_group = [[0, 1], [2, 3], [4, 5], [6, 7]]
        tp_group_idx = int(self.rank / 2)
        self.assertEqual(mesh_3d["TP"].mesh.tolist(), tp_group[tp_group_idx])

        shard_group = [[0, 2], [1, 3], [4, 6], [5, 7]]
        shard_group_idx = self.rank % 2 + self.rank // 4 * 2
        self.assertEqual(mesh_3d["Shard"].mesh.tolist(), shard_group[shard_group_idx])

        replicate_group = [[0, 4], [1, 5], [2, 6], [3, 7]]
        replicate_group_idx = self.rank % 4
        self.assertEqual(
            mesh_3d["Replicate"].mesh.tolist(), replicate_group[replicate_group_idx]
        )

        # We support both UX for nD slicing.
        # E.g. mesh_3d[["Replicate", "Shard"]] or mesh_3d["Replicate", "Shard"].
        hsdp_mesh_1 = mesh_3d[["Replicate", "Shard"]]
        hsdp_mesh_2 = mesh_3d["Replicate", "Shard"]
        hsdp_group = [[[0, 2], [4, 6]], [[1, 3], [5, 7]]]
        hsdp_group_idx = self.rank % 2
        self.assertEqual(hsdp_mesh_1.mesh.tolist(), hsdp_group[hsdp_group_idx])
        self.assertEqual(hsdp_mesh_2.mesh.tolist(), hsdp_group[hsdp_group_idx])
        self.assertEqual(hsdp_mesh_1, hsdp_mesh_2)

    @skipIfUnsupportMultiNPU(8)
    @with_comms
    def test_get_item_3d_noncontiguous_slicing(self):
        mesh_shape = (2, 2, 2)
        mesh_dim_names = ("dp", "pp", "cp")
        mesh_3d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        # Slice order simply decides which mesh_dim sits on which mesh_dim.
        # For dp_cp_mesh, cp mesh is the innermost dimension.
        dp_cp_mesh = mesh_3d["dp", "cp"]
        expected_mesh_tensor = (
            torch.tensor([[0, 1], [4, 5]], dtype=torch.int)
            if self.rank in (0, 1, 4, 5)
            else torch.tensor([[2, 3], [6, 7]], dtype=torch.int)
        )
        dp_local_rank = dp_cp_mesh.get_local_rank("dp")
        self.assertEqual(dp_cp_mesh.mesh, expected_mesh_tensor)
        cp_mesh = mesh_3d["cp"]
        # Check on the current dp_local_rank, whether the cp mesh tensor is the same.
        self.assertEqual(dp_cp_mesh.mesh[dp_local_rank], cp_mesh.mesh)

        with self.assertRaisesRegex(
            KeyError,
            "Invalid mesh_dim_names",
        ):
            cp_dp_mesh = mesh_3d["cp", "dp"]

    @skipIfUnsupportMultiNPU(8)
    @with_comms
    def test_reconstruct_mesh_with_flatten_dim(self):
        mesh_3d = init_device_mesh(
            self.device_type, (2, 2, 2), mesh_dim_names=("replicate", "shard", "cp")
        )
        shard_cp_mesh = mesh_3d["shard", "cp"]._flatten()
        hsdp_mesh = mesh_3d["replicate", "shard_cp"]
        expected_mesh_tensor = torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int
        )
        self.assertEqual(hsdp_mesh.mesh, expected_mesh_tensor)
        self.assertEqual(shard_cp_mesh.get_group(), mesh_3d["shard_cp"].get_group())
        self.assertEqual(
            shard_cp_mesh.get_group(), mesh_3d.get_group(mesh_dim="shard_cp")
        )

        mesh_3d = init_device_mesh(
            self.device_type, (2, 2, 2), mesh_dim_names=("dp", "cp", "tp")
        )
        dp_cp_mesh = mesh_3d["dp", "cp"]._flatten()
        spmd_mesh = mesh_3d["dp_cp", "tp"]
        expected_mesh_tensor = torch.tensor(
            [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int
        )
        self.assertEqual(spmd_mesh.mesh, expected_mesh_tensor)
        self.assertEqual(dp_cp_mesh.get_group(), mesh_3d["dp_cp"].get_group())
        self.assertEqual(dp_cp_mesh.get_group(), mesh_3d.get_group(mesh_dim="dp_cp"))


class TestMeshEnv(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_get_root_mesh(self):
        mesh_3d = init_device_mesh(
            self.device_type, (2, 1, 1), mesh_dim_names=("dp", "cp", "tp")
        )

        dp_cp_mesh = mesh_3d["dp", "cp"]
        dp_tp_mesh = mesh_3d["dp", "tp"]
        cp_tp_mesh = mesh_3d["cp", "tp"]
        dp_mesh = mesh_3d["dp"]
        cp_mesh = mesh_3d["cp"]
        tp_mesh = mesh_3d["tp"]
        self.assertEqual(_mesh_resources.get_root_mesh(dp_cp_mesh), mesh_3d)
        self.assertEqual(_mesh_resources.get_root_mesh(dp_tp_mesh), mesh_3d)
        self.assertEqual(_mesh_resources.get_root_mesh(cp_tp_mesh), mesh_3d)
        self.assertEqual(_mesh_resources.get_root_mesh(dp_mesh), mesh_3d)
        self.assertEqual(_mesh_resources.get_root_mesh(cp_mesh), mesh_3d)
        self.assertEqual(_mesh_resources.get_root_mesh(tp_mesh), mesh_3d)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_get_root_mesh_dim_exist(self):
        mesh_shape = (2, self.world_size // 2)
        mesh_dim_names = ("DP", "TP")
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        self.assertEqual(_mesh_resources.get_root_mesh_dim(mesh_2d["DP"]), 0)
        self.assertEqual(_mesh_resources.get_root_mesh_dim(mesh_2d["TP"]), 1)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_get_root_mesh_dim_not_exist(self):
        mesh_shape = (self.world_size,)
        mesh = init_device_mesh(self.device_type, mesh_shape)

        self.assertEqual(_mesh_resources.get_root_mesh_dim(mesh), None)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_get_mesh_dim_by_name(self):
        mesh_shape = (2, self.world_size // 2)
        mesh_dim_names = ("DP", "TP")
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        self.assertEqual(_mesh_resources.get_mesh_dim_by_name(mesh_2d, "DP"), 0)
        self.assertEqual(_mesh_resources.get_mesh_dim_by_name(mesh_2d, "TP"), 1)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_get_all_submeshes(self):
        mesh_2d = init_device_mesh(
            self.device_type, (1, 2), mesh_dim_names=("replicate", "shard")
        )
        all_submeshes = _mesh_resources._get_all_submeshes(mesh_2d, "replicate")
        self.assertEqual(len(all_submeshes), 2)
        self.assertEqual(
            all(submesh.mesh.numel() == 1 for submesh in all_submeshes), True
        )


class DeviceMeshCollectiveTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_broadcast_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank
        mesh_broadcast(local_tensor, mesh, mesh_dim=0)
        self.assertEqual(local_tensor, torch.zeros(3, 3))

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_scatter_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        scatter_tensor_shape = [3, 3, 3]
        len_scatter_tensor_shape = len(scatter_tensor_shape)
        for scatter_dim in range(len_scatter_tensor_shape):
            shard_placement = Shard(scatter_dim)
            scatter_tensor_shape[scatter_dim] *= self.world_size
            # make the random seed same across rank
            torch.manual_seed(0)
            global_tensor = torch.randn(scatter_tensor_shape, device=self.device_type)
            splitted_list, _ = shard_placement._split_tensor(
                global_tensor, mesh.size(), with_padding=True, contiguous=True
            )
            recv_tensor = torch.empty_like(splitted_list[mesh.get_rank()])
            # scatter on dim > 0 would generate non-contiguous tensor, verify that works
            mesh_scatter(recv_tensor, splitted_list, mesh, mesh_dim=0)
            self.assertEqual(recv_tensor, splitted_list[mesh.get_rank()])

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_scatter_uneven(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        my_rank = device_mesh.get_rank()
        tensor_to_split = torch.randn(
            device_mesh.size() + 3, device_mesh.size() + 1, device=self.device_type
        )

        for shard_dim in range(tensor_to_split.ndim):
            shard_placement = Shard(shard_dim)

            tensor_to_scatter = tensor_to_split.clone()
            tensor_splitted_list = list(
                torch.chunk(tensor_to_split, self.world_size, dim=shard_dim)
            )
            for _ in range(self.world_size - len(tensor_splitted_list)):
                tensor_splitted_list.append(torch.tensor([], device=self.device_type))

            padded_tensor_list, pad_sizes = shard_placement._split_tensor(
                tensor_to_scatter,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )

            scattered_tensor = torch.empty_like(padded_tensor_list[my_rank])
            mesh_scatter(scattered_tensor, padded_tensor_list, device_mesh, mesh_dim=0)

            if pad_sizes[my_rank] != 0:
                scattered_tensor = unpad_tensor(
                    scattered_tensor, shard_dim, pad_sizes[my_rank]
                )

            if scattered_tensor.numel() == 0:
                # We need to check numel() instead of size if a tensor is ([]) after unpadding,
                # since the size could be ([0, 8]) after unpadding.
                self.assertEqual(
                    scattered_tensor.numel(), tensor_splitted_list[my_rank].numel()
                )
            else:
                self.assertEqual(
                    scattered_tensor.size(), tensor_splitted_list[my_rank].size()
                )
                self.assertEqual(scattered_tensor, tensor_splitted_list[my_rank])

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_all_gather_uneven(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        my_rank = device_mesh.get_rank()
        tensor_to_split = torch.ones(
            device_mesh.size() + 3,
            device_mesh.size() + 1,
            device=self.device_type,
        )

        for shard_dim in range(tensor_to_split.ndim):
            shard_placement = Shard(shard_dim)
            tensor_padded_list, pad_sizes = shard_placement._split_tensor(
                tensor_to_split,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )
            local_tensor = tensor_padded_list[my_rank]
            big_tensor = funcol.all_gather_tensor(
                local_tensor, gather_dim=shard_dim, group=(device_mesh, 0)
            )
            big_tensor_chunks = list(
                torch.chunk(big_tensor, device_mesh.size(), dim=shard_dim)
            )
            unpadded_list = [
                (
                    unpad_tensor(big_tensor, shard_dim, pad_sizes[i])
                    if pad_sizes[i] > 0
                    else big_tensor
                )
                for i, big_tensor in enumerate(big_tensor_chunks)
            ]
            all_gathered_tensor = torch.cat(unpadded_list, dim=shard_dim)

            self.assertEqual(all_gathered_tensor.size(), tensor_to_split.size())
            self.assertEqual(all_gathered_tensor, tensor_to_split)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_reduce_scatter_contiguous(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        my_rank = device_mesh.get_rank()

        # Init the tensor
        step = self.world_size * 2
        total_elem = step**2
        tensor = torch.arange(0, total_elem).view(step, -1).to(device=self.device_type)
        tensor = tensor * (my_rank + 1)

        # Get non-contiguous tensor by slicing
        tensor_to_reduce = tensor[::2, :2]
        tensor_contiguous = tensor_to_reduce.clone().contiguous()

        # Partial to Shard to trigger reduce_scatter
        tensor_to_reduce = DTensor.from_local(
            tensor_to_reduce, device_mesh, [_Partial()]
        )
        tensor_contiguous = DTensor.from_local(
            tensor_contiguous, device_mesh, [_Partial()]
        )
        new_tensor = tensor_to_reduce.redistribute(device_mesh, [Shard(0)])
        new_tensor_contiguous = tensor_contiguous.redistribute(device_mesh, [Shard(0)])

        # The output for contiguous and non-contiguous tensors of the same value
        # should return the same reducescatter value.
        new_tensor_local = new_tensor._local_tensor
        new_tensor_contiguous_local = new_tensor_contiguous._local_tensor
        self.assertEqual(new_tensor_local, new_tensor_contiguous_local)
        self.assertEqual(list(new_tensor_local.size()), [1, 2])

        # Check the reduce numerical value
        sum_base = (1 + self.world_size) * self.world_size / 2
        first_elem = my_rank * sum_base * step * 2
        expected_tensor = torch.tensor(
            [[first_elem, first_elem + sum_base]],
            dtype=new_tensor_local.dtype,
            device=self.device_type,
        )
        self.assertEqual(new_tensor_local, expected_tensor)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_reduce_scatter_uneven(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        my_rank = device_mesh.get_rank()
        tensor_to_split = (
            torch.ones(
                device_mesh.size() + 3,
                device_mesh.size() + 1,
                device=self.device_type,
            )
            * self.rank
        )

        for shard_dim in range(tensor_to_split.ndim):
            shard_placement = Shard(shard_dim)
            tensor_to_scatter = tensor_to_split.clone()

            tensor_splitted_list = list(
                torch.chunk(tensor_to_split, self.world_size, dim=shard_dim)
            )
            for _ in range(self.world_size - len(tensor_splitted_list)):
                tensor_splitted_list.append(torch.tensor([], device=self.device_type))

            padded_tensor_list, pad_sizes = shard_placement._split_tensor(
                tensor_to_scatter,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )

            tensor_to_reduce = torch.cat(padded_tensor_list, shard_dim)

            res_num = ((0 + self.world_size - 1) * self.world_size) / 2

            scattered_tensor = funcol.reduce_scatter_tensor(
                tensor_to_reduce,
                reduceOp="sum",
                scatter_dim=shard_dim,
                group=(device_mesh, 0),
            )

            # unpad scattered_tensor
            if pad_sizes[my_rank] > 0:
                scattered_tensor = unpad_tensor(
                    scattered_tensor, shard_dim, pad_sizes[my_rank]
                )

            if scattered_tensor.numel() == 0:
                # We need to check numel() instead of size if a tensor is ([]) after unpadding,
                # since the size could be ([0, 8]) after unpadding.
                self.assertEqual(
                    scattered_tensor.numel(), tensor_splitted_list[my_rank].numel()
                )
            else:
                self.assertEqual(
                    scattered_tensor.size(), tensor_splitted_list[my_rank].size()
                )
                self.assertEqual(
                    scattered_tensor,
                    torch.ones_like(tensor_splitted_list[my_rank]) * res_num,
                )

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_broadcast_nd(self):
        mesh_tensor = torch.arange(2).reshape(2, 1, 1)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # check all dim groups
        dim_to_subgroups = mesh.get_all_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            cloned_local_tensor = local_tensor.clone()
            mesh_broadcast(cloned_local_tensor, mesh, mesh_dim=dim)
            res_num = global_ranks[0]
            self.assertEqual(cloned_local_tensor, torch.ones(3, 3) * res_num)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_scatter_nd(self):
        mesh_tensor = torch.arange(2).reshape(2, 1, 1)
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_all_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            scattered_tensors = [
                torch.ones(3, 3, device=self.device_type) * global_rank
                for global_rank in global_ranks
            ]
            received_tensor = torch.empty_like(
                scattered_tensors[mesh.get_coordinate()[dim]]
            )
            mesh_scatter(received_tensor, scattered_tensors, mesh, mesh_dim=dim)
            self.assertEqual(received_tensor, torch.ones(3, 3) * self.rank)


if __name__ == "__main__":
    run_tests()