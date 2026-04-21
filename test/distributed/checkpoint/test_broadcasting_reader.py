"""
Add validation cases for BroadcastingTorchSaveReader APIs:
1. pytorch/test/distributed/checkpoint/test_format_utils.py from PyTorch community lacks sufficient API validations, so this file is added.
2. This file validates the following APIs:
   torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.reset
   torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.set_up_storage_reader
   torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.validate_checkpoint_id
   (extendable)
"""

import os
import tempfile
from unittest.mock import patch

import torch
import torch_npu
from torch.distributed.checkpoint.format_utils import BroadcastingTorchSaveReader
from torch.testing._internal.common_utils import TestCase, run_tests


class TestBroadcastingTorchSaveReader(TestCase):
    """Independent unit tests for pure logic APIs of BroadcastingTorchSaveReader.

    This test uses mocks for distributed functions, allowing full validation
    of all APIs without starting a process group.
    """

    def setUp(self):
        """Runs before each test method: creates a valid temporary checkpoint file."""
        fd, self.temp_file_path = tempfile.mkstemp(suffix=".pt", dir=".")
        os.close(fd)

        tensor = torch.tensor([1, 2, 3], device="npu")
        torch.save({"dummy": tensor}, self.temp_file_path)
        self.valid_checkpoint_id = self.temp_file_path

    def tearDown(self):
        """Runs after each test method: deletes the temporary file to clean up."""
        if os.path.exists(self.valid_checkpoint_id):
            os.unlink(self.valid_checkpoint_id)

            self.assertFalse(
                os.path.exists(self.valid_checkpoint_id),
                f"tearDown failed to cleanup file: {self.valid_checkpoint_id}"
            )

    def test_validate_checkpoint_id_valid_path(self):
        """Passing a path to an existing file should return True."""
        result = BroadcastingTorchSaveReader.validate_checkpoint_id(self.valid_checkpoint_id)
        self.assertTrue(result, "valid checkpoint file should return True")

    def test_validate_checkpoint_id_invalid_path(self):
        """Passing a path to a non-existent file should return False."""
        non_existent = "does_not_exist_12345.pt"
        result = BroadcastingTorchSaveReader.validate_checkpoint_id(non_existent)
        self.assertFalse(result, "non-existent file should return False")

    def test_reset_updates_checkpoint_id(self):
        """Verify that the reset method correctly updates the internal checkpoint_id."""
        reader = BroadcastingTorchSaveReader(checkpoint_id="old_path.pt")
        new_path = "new_path.pt"
        reader.reset(new_path)
        self.assertEqual(reader.checkpoint_id, new_path, "reset should update checkpoint_id")

    def test_reset_none(self):
        """Verify that reset with None correctly sets checkpoint_id to None."""
        reader = BroadcastingTorchSaveReader(checkpoint_id="abc")
        reader.reset(None)
        self.assertIsNone(reader.checkpoint_id)

    class DummyMetadata:
        """Placeholder Metadata class used for testing."""
        pass

    @patch("torch.distributed.get_rank")
    def test_set_up_storage_reader_sets_coordinator_flag(self, mock_get_rank):
        """Verify that set_up_storage_reader correctly sets the is_coordinator attribute."""
        mock_get_rank.return_value = 0

        reader = BroadcastingTorchSaveReader(checkpoint_id=self.valid_checkpoint_id)
        metadata = self.DummyMetadata()

        reader.set_up_storage_reader(metadata, is_coordinator=True)
        self.assertTrue(reader.is_coordinator, "is_coordinator should be set to True")

        reader.set_up_storage_reader(metadata, is_coordinator=False)
        self.assertFalse(reader.is_coordinator, "is_coordinator should be set to False")

    @patch("torch.distributed.get_rank")
    def test_set_up_storage_reader_coordinator_rank_mismatch_raises(self, mock_get_rank):
        """Verify that an AssertionError is raised when is_coordinator=True
        but the current rank does not match coordinator_rank."""
        mock_get_rank.return_value = 1

        reader = BroadcastingTorchSaveReader(checkpoint_id=self.valid_checkpoint_id, coordinator_rank=0)
        metadata = self.DummyMetadata()

        with self.assertRaises(AssertionError, msg="Should raise AssertionError when coordinator rank mismatches"):
            reader.set_up_storage_reader(metadata, is_coordinator=True)

    def test_set_up_storage_reader_requires_checkpoint_id(self):
        """Verify that an AssertionError is raised when checkpoint_id is None."""
        reader = BroadcastingTorchSaveReader(checkpoint_id=None)
        metadata = self.DummyMetadata()

        with patch("torch.distributed.get_rank", return_value=0):
            with self.assertRaises(AssertionError, msg="checkpoint_id not set should raise AssertionError"):
                reader.set_up_storage_reader(metadata, is_coordinator=True)


if __name__ == "__main__":
    run_tests()