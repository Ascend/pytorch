"""
Add validation cases for torch.distributed.checkpoint.planner APIs on NPU:
1. PyTorch community tests lack sufficient validation for WriteItem.tensor_storage_size, so this file is added.
2. This file validates torch.distributed.checkpoint.planner.WriteItem.tensor_storage_size.
"""

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import (
    TensorWriteData,
    WriteItem,
    WriteItemType,
)
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class TestPlannerAPI(TestCase):

    def _make_tensor_write_item(self, tensor, write_item_type):
        tensor_data = TensorWriteData(
            chunk=ChunkStorageMetadata(
                offsets=torch.Size([0] * tensor.dim()),
                sizes=tensor.size(),
            ),
            properties=TensorProperties.create_from_tensor(tensor),
            size=tensor.size(),
        )

        return WriteItem(
            index=MetadataIndex("tensor"),
            type=write_item_type,
            tensor_data=tensor_data,
        )

    def test_write_item_tensor_storage_size_for_tensor(self):
        for dtype in (torch.float32, torch.float16, torch.int8):
            tensor = torch.empty((2, 3), dtype=dtype).to(device_type)
            write_item = self._make_tensor_write_item(
                tensor,
                WriteItemType.TENSOR,
            )

            expected_size = tensor.numel() * tensor.element_size()
            self.assertEqual(write_item.tensor_storage_size(), expected_size)

    def test_write_item_tensor_storage_size_for_shard(self):
        tensor = torch.empty((2, 3), dtype=torch.float32).to(device_type)
        write_item = self._make_tensor_write_item(
            tensor,
            WriteItemType.SHARD,
        )

        expected_size = tensor.numel() * tensor.element_size()
        self.assertEqual(write_item.tensor_storage_size(), expected_size)

    def test_write_item_tensor_storage_size_for_non_tensor(self):
        write_item = WriteItem(
            index=MetadataIndex("bytes"),
            type=WriteItemType.BYTE_IO,
        )

        self.assertIsNone(write_item.tensor_storage_size())


if __name__ == "__main__":
    run_tests()
