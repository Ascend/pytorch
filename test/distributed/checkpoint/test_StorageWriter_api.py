"""
PyTorch community lacks some torch.distributed.checkpoint.StorageWriter APIs validation, so this file is added.

This file validates the following apis:
torch.distributed.checkpoint.StorageWriter.storage_meta
(extendable)
"""

import os
import tempfile
import uuid

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.metadata import Metadata, StorageMeta
from torch.distributed.checkpoint.storage import StorageWriter, WriteResult
from torch.futures import Future
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
)

from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU, with_comms
from torch_npu.testing.testcase import run_tests, TestCase


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class _MinimalStorageWriter(StorageWriter):
    """Minimal subclass implementing only abstractmethods to exercise the base
    default storage_meta() which returns None.
    """

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        return None

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        return None

    def prepare_local_plan(self, plan):
        return plan

    def prepare_global_plan(self, plans):
        return plans

    def write_data(self, plan, planner) -> Future[list[WriteResult]]:
        fut: Future[list[WriteResult]] = Future()
        fut.set_result([])
        return fut

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        return None

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return True


class TestStorageMetaSingleProcess(TestCase):
    """Single-process contract validation: base default, FileSystemWriter field
    semantics, post-save validity, empty-state-dict boundary, reset lifecycle,
    and SavePlanner propagation.
    """

    # --------------------------------------------------------------------- #
    # Base-class default                                                      #
    # --------------------------------------------------------------------- #
    def test_base_storage_meta_callable_and_returns_none(self):
        self.assertTrue(callable(StorageWriter.storage_meta))
        writer = _MinimalStorageWriter()
        self.assertIsNone(writer.storage_meta())

    # --------------------------------------------------------------------- #
    # FileSystemWriter field semantics                                        #
    # --------------------------------------------------------------------- #
    def _assert_storage_meta_fields(self, writer: FileSystemWriter, expected_dir: str):
        meta = writer.storage_meta()
        self.assertIsInstance(meta, StorageMeta)
        self.assertEqual(str(meta.checkpoint_id), expected_dir)
        self.assertIsInstance(meta.modules, list)
        self.assertEqual(meta.modules, [])
        self.assertIsInstance(uuid.UUID(meta.save_id), uuid.UUID)

    def test_filesystem_writer_storage_meta_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = FileSystemWriter(temp_dir)
            self._assert_storage_meta_fields(writer, temp_dir)

    def test_storage_meta_after_reset(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            writer = FileSystemWriter(d1)
            save_id_before = writer.storage_meta().save_id

            writer.reset(checkpoint_id=d2)
            meta_after = writer.storage_meta()

            self.assertEqual(str(meta_after.checkpoint_id), d2)
            self.assertNotEqual(save_id_before, meta_after.save_id)

    # --------------------------------------------------------------------- #
    # Save integration                                                        #
    # --------------------------------------------------------------------- #
    def test_storage_meta_valid_after_save(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dict = {"weight": torch.randn(4, 4).to(device_type)}
            writer = FileSystemWriter(temp_dir)
            dcp.save(state_dict, storage_writer=writer, no_dist=True)
            self._assert_storage_meta_fields(writer, temp_dir)

    def test_storage_meta_empty_state_dict(self):
        """Boundary: writer.storage_meta() must remain valid even when no tensors
        are written (empty state_dict).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = FileSystemWriter(temp_dir)
            dcp.save({}, storage_writer=writer, no_dist=True)
            self._assert_storage_meta_fields(writer, temp_dir)

    def test_storage_meta_reset_between_saves(self):
        """save -> reset(new dir) -> save: save_id must change to guarantee
        uniqueness across distinct save operations, preventing metadata collisions
        in distributed checkpointing scenarios.
        """
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            writer = FileSystemWriter(d1)
            dcp.save(
                {"w": torch.randn(4, 4).to(device_type)},
                storage_writer=writer,
                no_dist=True,
            )
            save_id_1 = writer.storage_meta().save_id

            writer.reset(checkpoint_id=d2)
            dcp.save(
                {"w": torch.randn(4, 4).to(device_type)},
                storage_writer=writer,
                no_dist=True,
            )
            save_id_2 = writer.storage_meta().save_id

            self.assertNotEqual(save_id_1, save_id_2)

    # --------------------------------------------------------------------- #
    # SavePlanner propagation                                                 #
    # --------------------------------------------------------------------- #
    def test_storage_meta_passed_to_planner(self):
        captured = {}

        class _CapturingPlanner(DefaultSavePlanner):
            def set_up_planner(
                self, state_dict, storage_meta=None, is_coordinator=False
            ):
                captured["meta"] = storage_meta
                super().set_up_planner(state_dict, storage_meta, is_coordinator)

        with tempfile.TemporaryDirectory() as temp_dir:
            state_dict = {"w": torch.randn(4, 4).to(device_type)}
            writer = FileSystemWriter(temp_dir)
            expected = writer.storage_meta()

            dcp.save(
                state_dict,
                storage_writer=writer,
                planner=_CapturingPlanner(),
                no_dist=True,
            )

            self.assertIsInstance(captured.get("meta"), StorageMeta)
            self.assertEqual(
                str(captured["meta"].checkpoint_id),
                str(expected.checkpoint_id),
            )
            self.assertEqual(captured["meta"].save_id, expected.save_id)


class TestStorageMetaDistributed(ShardedTensorTestBase):
    """Multi-NPU contract: checkpoint_id must be consistent across ranks,
    save_id must remain unique per writer instance, and storage_meta must be
    obtainable directly from the writer after distributed save.

    NOTE: We assert writer.storage_meta() rather than Metadata.storage_meta
    because PyTorch 2.9+ leaves the latter None on the no_dist=True path
    (community Issue #177887). The writer-side API contract is the stable
    surface for validation.
    """

    def tearDown(self):
        super().tearDown()
        self.destroy_pg()

    def destroy_pg(self) -> None:
        """Best-effort cleanup: swallow exceptions to avoid masking the real
        test result when the process group is already in a bad state.
        """
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    @property
    def world_size(self) -> int:
        return 2

    def _shared_temp_dir(self):
        """Yield a single checkpoint directory visible to all ranks.
        Rank 0 creates the directory and broadcasts its path; all ranks
        synchronize on teardown so rank 0 can safely remove it.
        """

        class _SharedTempDirContext:
            def __init__(self):
                self._temp_dir_ctx = None
                self.temp_dir = ""

            def __enter__(self):
                if dist.get_rank() == 0:
                    self._temp_dir_ctx = tempfile.TemporaryDirectory()
                    self.temp_dir = self._temp_dir_ctx.__enter__()
                object_list = [self.temp_dir]
                dist.broadcast_object_list(object_list, src=0)
                self.temp_dir = object_list[0]
                return self.temp_dir

            def __exit__(self, exc_type, exc, tb):
                dist.barrier()
                if dist.get_rank() == 0 and self._temp_dir_ctx is not None:
                    self._temp_dir_ctx.__exit__(exc_type, exc, tb)
                return False

        return _SharedTempDirContext()

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_storage_meta_checkpoint_id_consistency_across_ranks(self) -> None:
        with self._shared_temp_dir() as temp_dir:
            writer = FileSystemWriter(temp_dir)
            dcp.save({"t": torch.randn(4, 4).to(device_type)}, storage_writer=writer)

            local_id = str(writer.storage_meta().checkpoint_id)
            gathered: list[str | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local_id)
            for cid in gathered:
                self.assertEqual(cid, gathered[0])

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_storage_meta_save_id_changes_after_reset_distributed(self) -> None:
        with self._shared_temp_dir() as temp_dir:
            writer = FileSystemWriter(temp_dir)
            dcp.save({"t": torch.randn(4, 4).to(device_type)}, storage_writer=writer)
            save_id_1 = writer.storage_meta().save_id

            dist.barrier()
            writer.reset(checkpoint_id=temp_dir)
            dcp.save({"t": torch.randn(4, 4).to(device_type)}, storage_writer=writer)
            save_id_2 = writer.storage_meta().save_id

            self.assertNotEqual(save_id_1, save_id_2)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_storage_meta_returned_in_distributed_save(self) -> None:
        with self._shared_temp_dir() as temp_dir:
            writer = FileSystemWriter(temp_dir)
            dcp.save(
                {f"rank_{dist.get_rank()}": torch.randn(4, 4).to(device_type)},
                storage_writer=writer,
            )

            meta = writer.storage_meta()
            self.assertIsInstance(meta, StorageMeta)
            self.assertIsInstance(uuid.UUID(meta.save_id), uuid.UUID)


if __name__ == "__main__":
    run_tests()
