"""
1. PyTorch community lacks direct validation cases for some
   torch.distributed.checkpoint.staging APIs, so this file is added.

2. This file validates the following APIs:
   torch.distributed.checkpoint.staging.AsyncStager
   torch.distributed.checkpoint.staging.AsyncStager.should_synchronize_after_execute
   torch.distributed.checkpoint.staging.AsyncStager.stage
   torch.distributed.checkpoint.staging.AsyncStager.synchronize_staging
   torch.distributed.checkpoint.staging.BlockingAsyncStager
   torch.distributed.checkpoint.staging.BlockingAsyncStager.stage
   torch.distributed.checkpoint.staging.BlockingAsyncStager.synchronize_staging
   (extendable)
"""

import tempfile

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.staging import AsyncStager, BlockingAsyncStager
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType

from torch_npu.testing.testcase import run_tests, TestCase


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestStatefulObj:
    def __init__(self, tensor):
        self.tensor = tensor

    def state_dict(self):
        return {"tensor": self.tensor}

    def load_state_dict(self, state_dict):
        self.tensor.copy_(state_dict["tensor"])


class SyncTrackingFileSystemWriter(FileSystemWriter):
    _synchronize_after_execute = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.synchronize_staging_called = False

    def synchronize_staging(self):
        self.synchronize_staging_called = True


class TestCheckpointStagingApi(TestCase):
    def test_async_stager_protocol_defaults(self):
        class UnimplementedStager:
            _synchronize_after_execute = True

            @property
            def should_synchronize_after_execute(self):
                return AsyncStager.should_synchronize_after_execute.fget(self)

            def stage(self, state_dict):
                return AsyncStager.stage(self, state_dict)

            def synchronize_staging(self):
                return AsyncStager.synchronize_staging(self)

            def close(self):
                return None

        stager = UnimplementedStager()

        self.assertIsInstance(stager, AsyncStager)
        self.assertTrue(stager.should_synchronize_after_execute)
        with self.assertRaisesRegex(NotImplementedError, "must implement stage method"):
            stager.stage({"tensor": torch.ones(2).to(device_type)})
        self.assertIsNone(stager.synchronize_staging())

    def test_blocking_async_stager_stage_npu_tensor_to_cpu_snapshot(self):
        state_dict = {
            "weight": torch.arange(12, dtype=torch.float32)
            .reshape(3, 4)
            .to(device_type),
            "nested": {
                "bias": torch.ones(4, dtype=torch.float32).to(device_type),
            },
        }
        original_weight = state_dict["weight"].cpu().clone()

        stager = BlockingAsyncStager(cache_staged_state_dict=False)
        staged_state_dict = stager.stage(state_dict)

        self.assertEqual("cpu", staged_state_dict["weight"].device.type)
        self.assertEqual("cpu", staged_state_dict["nested"]["bias"].device.type)
        self.assertEqual(original_weight, staged_state_dict["weight"])

        state_dict["weight"].add_(100)
        self.assertEqual(original_weight, staged_state_dict["weight"])
        self.assertNotEqual(state_dict["weight"].cpu(), staged_state_dict["weight"])

    def test_blocking_async_stager_cached_reuses_cpu_buffer(self):
        stager = BlockingAsyncStager(cache_staged_state_dict=True)

        first_state_dict = {
            "tensor": torch.ones(8, dtype=torch.float32).to(device_type)
        }
        first_staged = stager.stage(first_state_dict)
        first_data_ptr = first_staged["tensor"].data_ptr()

        second_state_dict = {
            "tensor": torch.full((8,), 3.0, dtype=torch.float32).to(device_type),
        }
        second_staged = stager.stage(second_state_dict)
        second_data_ptr = second_staged["tensor"].data_ptr()

        self.assertEqual(first_data_ptr, second_data_ptr)
        self.assertEqual(second_state_dict["tensor"].cpu(), second_staged["tensor"])

    def test_blocking_async_stager_sync_noop_and_property(self):
        stager = BlockingAsyncStager()

        self.assertIsInstance(stager, AsyncStager)
        self.assertFalse(stager.should_synchronize_after_execute)
        self.assertIsNone(stager.synchronize_staging())

    def test_filesystem_writer_stage_sets_copy_ahead_zero(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            writer = FileSystemWriter(
                checkpoint_dir,
                per_thread_copy_ahead=1024,
            )
            staged_state_dict = writer.stage(
                {"tensor": torch.ones(4, dtype=torch.float32).to(device_type)}
            )

            self.assertEqual(0, writer.per_thread_copy_ahead)
            self.assertEqual("cpu", staged_state_dict["tensor"].device.type)
            self.assertEqual(
                torch.ones(4, dtype=torch.float32),
                staged_state_dict["tensor"],
            )

    def test_async_save_npu_state_dict_with_filesystem_writer(self):
        for cache_staged_state_dict in (False, True):
            with self.subTest(cache_staged_state_dict=cache_staged_state_dict):
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    source_state_dict = {
                        "tensor": torch.arange(8, dtype=torch.float32).to(device_type),
                    }
                    writer = FileSystemWriter(
                        checkpoint_dir,
                        cache_staged_state_dict=cache_staged_state_dict,
                    )

                    future = dcp.async_save(
                        source_state_dict,
                        storage_writer=writer,
                        async_checkpointer_type=AsyncCheckpointerType.THREAD,
                    )
                    future.result()

                    loaded_state_dict = {
                        "tensor": torch.zeros(8, dtype=torch.float32).to(device_type),
                    }
                    dcp.load(loaded_state_dict, checkpoint_id=checkpoint_dir)

                    self.assertEqual(
                        source_state_dict["tensor"].cpu(),
                        loaded_state_dict["tensor"].cpu(),
                    )

    def test_async_save_converts_stateful_object(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            source_obj = TestStatefulObj(
                torch.arange(4, dtype=torch.float32).to(device_type)
            )

            future = dcp.async_save(
                {"obj": source_obj},
                storage_writer=FileSystemWriter(checkpoint_dir),
                async_checkpointer_type=AsyncCheckpointerType.THREAD,
            )
            future.result()

            loaded_obj = TestStatefulObj(
                torch.zeros(4, dtype=torch.float32).to(device_type)
            )
            dcp.load({"obj": loaded_obj}, checkpoint_id=checkpoint_dir)

            self.assertEqual(source_obj.tensor.cpu(), loaded_obj.tensor.cpu())

    def test_async_save_calls_synchronize_when_stager_requests_it(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            writer = SyncTrackingFileSystemWriter(checkpoint_dir)

            future = dcp.async_save(
                {"tensor": torch.ones(4, dtype=torch.float32).to(device_type)},
                storage_writer=writer,
                async_checkpointer_type=AsyncCheckpointerType.THREAD,
            )

            self.assertTrue(writer.synchronize_staging_called)
            future.result()


if __name__ == "__main__":
    run_tests()
