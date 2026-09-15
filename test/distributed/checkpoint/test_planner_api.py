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
Add validation cases for torch.distributed.checkpoint.planner APIs on NPU:
1. PyTorch community tests lack sufficient validation for TensorWriteData and
   WriteItem.tensor_storage_size, so this file is added.
2. This file validates torch.distributed.checkpoint.planner.TensorWriteData and
   torch.distributed.checkpoint.planner.WriteItem.tensor_storage_size.
"""

import dataclasses

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


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


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

    def _make_tensor_write_data(self, tensor):
        return TensorWriteData(
            chunk=ChunkStorageMetadata(
                offsets=torch.Size([0] * tensor.dim()),
                sizes=tensor.size(),
            ),
            properties=TensorProperties.create_from_tensor(tensor),
            size=tensor.size(),
        )

    def test_tensor_write_data_fields(self):
        for dtype in (torch.float32, torch.float16, torch.int8):
            tensor = torch.empty((2, 3), dtype=dtype).to(device_type)
            tensor_data = self._make_tensor_write_data(tensor)

            self.assertEqual(tensor_data.chunk.offsets, torch.Size([0, 0]))
            self.assertEqual(tensor_data.chunk.sizes, torch.Size([2, 3]))
            self.assertEqual(
                tensor_data.properties,
                TensorProperties.create_from_tensor(tensor),
            )
            self.assertEqual(tensor_data.size, torch.Size([2, 3]))

    def test_tensor_write_data_is_frozen(self):
        tensor = torch.empty((2, 3), dtype=torch.float32).to(device_type)
        tensor_data = self._make_tensor_write_data(tensor)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            tensor_data.size = torch.Size([3, 2])

    def test_tensor_write_data_equal(self):
        tensor = torch.empty((2, 3), dtype=torch.float32).to(device_type)
        tensor_data = self._make_tensor_write_data(tensor)
        tensor_data_2 = self._make_tensor_write_data(tensor)

        self.assertEqual(tensor_data, tensor_data_2)

    def test_tensor_write_data_unhashable(self):
        tensor = torch.empty((2, 3), dtype=torch.float32).to(device_type)
        tensor_data = self._make_tensor_write_data(tensor)

        with self.assertRaises(TypeError):
            hash(tensor_data)

    def test_tensor_write_data_invalid_args(self):
        with self.assertRaises(TypeError):
            TensorWriteData(
                chunk=ChunkStorageMetadata(
                    offsets=torch.Size([0, 0]),
                    sizes=torch.Size([2, 3]),
                ),
                properties=TensorProperties.create_from_tensor(
                    torch.empty((2, 3)).to(device_type)
                ),
            )


if __name__ == "__main__":
    run_tests()
