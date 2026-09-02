# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch.distributed.distributed_c10d.ProcessGroupMPI.create:

1. PyTorch community lacks direct Python test cases for ProcessGroupMPI.create,
   so this file is added.
2. This file validates the following apis:
torch.distributed.distributed_c10d.ProcessGroupMPI.create
(extendable)

Note: MPI backend requires the system to have an MPI implementation installed
(e.g., OpenMPI) and PyTorch to be compiled with MPI support. When MPI is not
available, most tests will be skipped, which is the expected behavior.
"""

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, run_tests


def _get_process_group_mpi():
    """Return ProcessGroupMPI when PyTorch is compiled with MPI support."""
    distributed_c10d = getattr(torch._C, "_distributed_c10d", None)
    return getattr(distributed_c10d, "ProcessGroupMPI", None)


def _is_mpi_usable():
    """Check whether MPI backend can actually be used."""
    return dist.is_available() and dist.is_mpi_available()


class TestProcessGroupMPIAvailability(TestCase):
    """Test MPI availability checks that work without MPI runtime."""

    def test_is_mpi_available_returns_bool(self):
        """Verify dist.is_mpi_available returns a boolean value."""
        result = dist.is_mpi_available()
        self.assertIsInstance(result, bool)

    def test_is_available_returns_bool(self):
        """Verify dist.is_available returns a boolean value."""
        result = dist.is_available()
        self.assertIsInstance(result, bool)

    def test_init_process_group_mpi_raises_when_not_available(self):
        """Verify init_process_group raises RuntimeError when MPI is not available."""
        if _is_mpi_usable():
            self.skipTest("MPI is available, cannot test error path")
        if not dist.is_available():
            self.skipTest("Distributed package is not available")

        with self.assertRaises(RuntimeError):
            dist.init_process_group(backend="mpi")

    def test_process_group_mpi_import_guard(self):
        """Verify ProcessGroupMPI import is guarded by availability."""
        ProcessGroupMPI = _get_process_group_mpi()
        if _is_mpi_usable():
            self.assertIsNotNone(ProcessGroupMPI)
        else:
            self.assertIsNone(ProcessGroupMPI)


class TestProcessGroupMPICreate(TestCase):
    """Test ProcessGroupMPI.create static factory method."""

    def _require_mpi(self):
        """Skip test if MPI is not usable."""
        ProcessGroupMPI = _get_process_group_mpi()
        if ProcessGroupMPI is None:
            self.skipTest("ProcessGroupMPI is not available (MPI not compiled)")
        if not _is_mpi_usable():
            self.skipTest("MPI backend is not available")
        return ProcessGroupMPI

    def _create_root_group(self):
        """Create an MPI group containing every rank in MPI_COMM_WORLD."""
        ProcessGroupMPI = self._require_mpi()
        pg = ProcessGroupMPI.create([])
        self.assertIsNotNone(pg)
        return ProcessGroupMPI, pg

    def test_create_rejects_invalid_argument_count(self):
        """Verify create rejects missing and extra arguments."""
        ProcessGroupMPI = self._require_mpi()
        with self.assertRaises(TypeError):
            ProcessGroupMPI.create()
        with self.assertRaises(TypeError):
            ProcessGroupMPI.create([], [])

    def test_create_rejects_invalid_ranks(self):
        """Verify create rejects invalid rank containers and elements."""
        ProcessGroupMPI = self._require_mpi()
        invalid_ranks = (None, 0, "0", [0.0], ["0"], [[0]])
        for ranks in invalid_ranks:
            with self.subTest(ranks=ranks):
                with self.assertRaises(TypeError):
                    ProcessGroupMPI.create(ranks)

    def test_create_returns_process_group(self):
        """Verify ProcessGroupMPI.create returns a ProcessGroupMPI instance."""
        _, pg = self._create_root_group()
        self.assertEqual(pg.name(), "mpi")

    def test_create_with_single_rank_properties(self):
        """Verify created process group has correct rank and size for single rank."""
        ProcessGroupMPI, root_pg = self._create_root_group()
        # MPI_Comm_create requires every rank to pass the same group definition.
        pg = ProcessGroupMPI.create([0])
        if root_pg.rank() == 0:
            self.assertIsNotNone(pg)
            self.assertEqual(pg.size(), 1)
            self.assertEqual(pg.rank(), 0)
        else:
            self.assertIsNone(pg)

    def test_create_with_world_size_ranks(self):
        """Verify ProcessGroupMPI.create works with full world size ranks."""
        ProcessGroupMPI, root_pg = self._create_root_group()
        ranks = list(range(root_pg.size()))
        pg = ProcessGroupMPI.create(ranks)
        self.assertIsNotNone(pg)
        self.assertEqual(pg.size(), root_pg.size())
        self.assertEqual(pg.rank(), root_pg.rank())

    def test_create_backend_name_is_mpi(self):
        """Verify the created process group has backend name 'mpi'."""
        _, pg = self._create_root_group()
        self.assertEqual(pg.name(), "mpi")

    def test_create_returns_none_for_non_member(self):
        """Verify ProcessGroupMPI.create returns None when current rank is not in the group."""
        ProcessGroupMPI, root_pg = self._create_root_group()
        world_size = root_pg.size()
        if world_size <= 1:
            self.skipTest("Need at least 2 processes to test non-member case")

        non_member_rank = world_size - 1
        member_ranks = list(range(non_member_rank))
        result = ProcessGroupMPI.create(member_ranks)
        if root_pg.rank() == non_member_rank:
            self.assertIsNone(result)
        else:
            self.assertIsNotNone(result)
            self.assertEqual(result.size(), len(member_ranks))


class TestProcessGroupMPIInitGroup(TestCase):
    """Test ProcessGroupMPI through dist.init_process_group with MPI backend."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mpi_usable = _is_mpi_usable()
        if cls.mpi_usable:
            dist.init_process_group(backend="mpi")
            cls.rank = dist.get_rank()
            cls.world_size = dist.get_world_size()

    def setUp(self):
        if not self.mpi_usable:
            self.skipTest("MPI backend is not available")

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDownClass()

    def test_get_backend_returns_mpi(self):
        """Verify dist.get_backend returns 'mpi' after MPI init."""
        backend = dist.get_backend()
        self.assertEqual(backend, "mpi")

    def test_get_rank(self):
        """Verify dist.get_rank returns the correct rank."""
        self.assertEqual(dist.get_rank(), self.rank)

    def test_get_world_size(self):
        """Verify dist.get_world_size returns the correct world size."""
        self.assertEqual(dist.get_world_size(), self.world_size)

    def test_is_initialized(self):
        """Verify dist.is_initialized returns True after MPI init."""
        self.assertTrue(dist.is_initialized())

    def test_allreduce(self):
        """Verify allreduce works with MPI backend."""
        tensor = torch.ones(4, 4)
        dist.all_reduce(tensor)
        expected = self.world_size
        self.assertTrue(tensor.equal(torch.full((4, 4), float(expected))))

    def test_broadcast(self):
        """Verify broadcast works with MPI backend."""
        if self.rank == 0:
            tensor = torch.ones(4, 4) * 42
        else:
            tensor = torch.zeros(4, 4)
        dist.broadcast(tensor, src=0)
        self.assertTrue(tensor.equal(torch.full((4, 4), 42.0)))

    def test_barrier(self):
        """Verify barrier works with MPI backend."""
        dist.barrier()


if __name__ == "__main__":
    run_tests()
