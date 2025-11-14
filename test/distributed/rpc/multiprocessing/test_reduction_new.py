import multiprocessing
from unittest.mock import patch

import torch
from torch.multiprocessing.reductions import (
    shared_cache,
    rebuild_storage_filename,
    rebuild_storage_empty,
    rebuild_storage_fd,
    StorageWeakRef,
    fd_id,
    rebuild_tensor,
    storage_from_cache,
)

import torch_npu
from torch_npu.multiprocessing.reductions import _npu_reduce_tensor, _npu_reduce_storage, _add_reductions_methods
import torch_npu.multiprocessing.reductions as reductions
from torch_npu.testing.testcase import TestCase, run_tests


class TestReduction(TestCase):
    def test_rebuild_npu_tensor_with_parameter_class(self):
        with patch('torch_npu.multiprocessing.reductions.storage_from_cache', return_value=None):
            with patch('torch_npu.npu._lazy_init') as mock_init:
                with patch('torch.UntypedStorage._new_shared_npu') as mock_new_shared:
                    mock_storage = torch.UntypedStorage(10, device="npu:0")
                    mock_new_shared.return_value = mock_storage

                    result = reductions.rebuild_npu_tensor(
                        torch.nn.parameter.Parameter,
                        (2, 3),
                        (3, 1),
                        0,
                        torch.UntypedStorage,
                        torch.float32,
                        "npu:0",
                        12345,
                        100,
                        0,
                        True,
                        None,
                        0,
                        None,
                        False
                    )
                    self.assertIsInstance(result, torch.nn.parameter.Parameter)
                    self.assertTrue(result.requires_grad)

    def test_npu_reduce_tensor_leaf_requires_grad(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True, device="npu:0")

        with patch.object(tensor._typed_storage(), '_share_npu_', return_value=(
                "npu:0", 12345, 100, 0, None, 0, None, False
        )):
            with patch.dict(reductions.shared_cache):
                try:
                    result = reductions._npu_reduce_tensor(tensor)
                    self.assertIsInstance(result, tuple)
                    self.assertEqual(len(result), 2)
                    self.assertEqual(result[0], reductions.rebuild_npu_tensor)
                except RuntimeError as e:
                    if "shareIpcHandle" in str(e):
                        self.skipTest("NPU IPC not supported in current environment")
                    raise

    def test_npu_reduce_storage_file_system_strategy(self):
        storage = torch.UntypedStorage(10, device="cpu")
        with patch('torch.multiprocessing.get_sharing_strategy', return_value="file_system"):
            with patch.object(storage, '_share_filename_cpu_', return_value=('filename', 12345)):
                with patch('torch.multiprocessing.reductions.rebuild_storage_filename',
                           return_value=lambda *args: None):
                    with patch.dict(reductions.shared_cache):
                        result = reductions._npu_reduce_storage(storage)
                        self.assertIsInstance(result, tuple)
                        self.assertEqual(len(result), 2)
                        self.assertEqual(result[0], reductions.rebuild_storage_filename)

    def test_npu_reduce_storage_npu_storage(self):
        storage = torch.UntypedStorage(10, device="npu:0")
        with self.assertRaises(RuntimeError):
            reductions._npu_reduce_storage(storage)

    def test_npu_reduce_tensor_non_leaf_requires_grad(self):
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = a * 2

        with self.assertRaises(RuntimeError):
            reductions._npu_reduce_tensor(b)

    def test_rebuild_npu_tensor_new_storage_creation(self):
        with patch('torch.multiprocessing.reductions.storage_from_cache', return_value=None):
            with patch('torch_npu.npu._lazy_init'):
                with patch.object(torch.UntypedStorage, '_new_shared_npu', return_value=torch.UntypedStorage(10)):
                    result = reductions.rebuild_npu_tensor(
                        torch.Tensor,
                        (2, 3),
                        (3, 1),
                        0,
                        torch.UntypedStorage,
                        torch.float32,
                        "npu:0",
                        "handle",
                        100,
                        0,
                        False,
                        "ref_handle",
                        0,
                        "event_handle",
                        True
                    )
                    self.assertIsInstance(result, torch.Tensor)
                    self.assertEqual(result.size(), torch.Size([2, 3]))

    def test_npu_reduce_storage_file_system_with_typed_storage(self):
        storage = torch.TypedStorage(10, dtype=torch.float32, device="cpu")
        with patch('torch.multiprocessing.get_sharing_strategy', return_value="file_system"):
            with patch.object(storage, '_share_filename_cpu_', return_value=("filename", 12345)):
                with patch('torch.multiprocessing.reductions.rebuild_storage_filename',
                           return_value=lambda *args: None):
                    with patch.dict(reductions.shared_cache):
                        result = reductions._npu_reduce_storage(storage)
                        self.assertIsInstance(result, tuple)
                        self.assertEqual(len(result), 2)
                        self.assertEqual(result[0], reductions.rebuild_storage_filename)


if __name__ == "__main__":
    run_tests()

