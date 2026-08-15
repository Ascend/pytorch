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

"""Add validation cases for torch.distributed.checkpoint.filesystem._write_item API on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates _write_item (extendable).
Note: _write_item asserts data.device == cpu, so tensors must remain on CPU.
"""


import inspect
import io
from unittest.mock import Mock

import torch
import torch.distributed.checkpoint.filesystem as filesystem
from torch.distributed.checkpoint.filesystem import (
    _write_item,
    _StorageInfo,
)

from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import WriteItemType
from torch.testing._internal.common_utils import run_tests, TestCase


def _call_write_item(stream, data, write_item, storage_key):
    params = inspect.signature(_write_item).parameters
    if "transforms" in params:
        return _write_item(
            filesystem._StorageWriterTransforms(),
            stream,
            data,
            write_item,
            storage_key,
            filesystem.SerializationFormat.TORCH_SAVE,
        )
    return _write_item(stream, data, write_item, storage_key)


class TestWriteItem(TestCase):
    """Test cases for _write_item function."""

    def _create_byte_write_item(self, index_value=0):
        """Create a mock WriteItem for BYTE_IO type."""
        item = Mock()
        item.type = WriteItemType.BYTE_IO
        item.index = MetadataIndex(index_value)
        return item

    def _create_tensor_write_item(self, index_value=0):
        """Create a mock WriteItem for TENSOR type."""
        item = Mock()
        item.type = WriteItemType.TENSOR
        item.index = MetadataIndex(index_value)
        return item

    def _assert_tensor_round_trip(self, stream, result, expected):
        """Deserialize the written tensor and verify its complete contents."""
        storage_info = result.storage_data
        original_position = stream.tell()
        stream.seek(storage_info.offset)
        payload = io.BytesIO(stream.read(storage_info.length))
        actual = torch.load(payload, weights_only=False)
        stream.seek(original_position)

        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual, expected)

    def test_write_byte_io_basic(self):
        """_write_item writes BytesIO data and returns correct WriteResult."""
        stream = io.BytesIO()
        data = io.BytesIO(b"hello world")
        write_item = self._create_byte_write_item()
        storage_key = "test_key_0"

        result = _call_write_item(stream, data, write_item, storage_key)

        self.assertEqual(result.index, write_item.index)
        self.assertEqual(result.size_in_bytes, 11)
        self.assertIsInstance(result.storage_data, _StorageInfo)
        self.assertEqual(result.storage_data.relative_path, storage_key)
        self.assertEqual(result.storage_data.offset, 0)
        self.assertEqual(result.storage_data.length, 11)

    def test_write_byte_io_empty(self):
        """_write_item handles empty BytesIO data."""
        stream = io.BytesIO()
        data = io.BytesIO(b"")
        write_item = self._create_byte_write_item()
        storage_key = "empty_key"

        result = _call_write_item(stream, data, write_item, storage_key)

        self.assertEqual(result.size_in_bytes, 0)
        self.assertEqual(result.storage_data.offset, 0)
        self.assertEqual(result.storage_data.length, 0)

    def test_write_tensor_basic(self):
        """_write_item writes tensor data and returns correct WriteResult."""
        stream = io.BytesIO()
        data = torch.tensor([1.0, 2.0, 3.0])
        write_item = self._create_tensor_write_item()
        storage_key = "tensor_key_0"

        result = _call_write_item(stream, data, write_item, storage_key)
        self._assert_tensor_round_trip(stream, result, data)

        self.assertEqual(result.index, write_item.index)
        self.assertGreater(result.size_in_bytes, 0)
        self.assertEqual(result.storage_data.relative_path, storage_key)
        self.assertEqual(result.storage_data.offset, 0)
        self.assertEqual(result.storage_data.length, result.size_in_bytes)

    def test_write_tensor_float32(self):
        """_write_item handles float32 tensor."""
        stream = io.BytesIO()
        data = torch.randn(10, dtype=torch.float32)
        write_item = self._create_tensor_write_item()
        storage_key = "f32_key"

        result = _call_write_item(stream, data, write_item, storage_key)
        self._assert_tensor_round_trip(stream, result, data)

        self.assertGreater(result.size_in_bytes, 0)
        self.assertEqual(result.storage_data.relative_path, storage_key)

    def test_write_tensor_int64(self):
        """_write_item handles int64 tensor."""
        stream = io.BytesIO()
        data = torch.tensor([1, 2, 3], dtype=torch.int64)
        write_item = self._create_tensor_write_item()
        storage_key = "int64_key"

        result = _call_write_item(stream, data, write_item, storage_key)
        self._assert_tensor_round_trip(stream, result, data)

        self.assertGreater(result.size_in_bytes, 0)

    def test_write_tensor_2d(self):
        """_write_item handles 2D tensor."""
        stream = io.BytesIO()
        data = torch.randn(3, 4, dtype=torch.float32)
        write_item = self._create_tensor_write_item()
        storage_key = "2d_key"

        result = _call_write_item(stream, data, write_item, storage_key)
        self._assert_tensor_round_trip(stream, result, data)

        self.assertGreater(result.size_in_bytes, 0)

    def test_write_tensor_scalar(self):
        """_write_item handles scalar (0-dim) tensor."""
        stream = io.BytesIO()
        data = torch.tensor(3.14)
        write_item = self._create_tensor_write_item()
        storage_key = "scalar_key"

        result = _call_write_item(stream, data, write_item, storage_key)
        self._assert_tensor_round_trip(stream, result, data)

        self.assertGreater(result.size_in_bytes, 0)

    def test_multiple_sequential_writes(self):
        """_write_item correctly tracks offset for sequential writes."""
        stream = io.BytesIO()

        data1 = io.BytesIO(b"first")
        item1 = self._create_byte_write_item(0)
        result1 = _call_write_item(stream, data1, item1, "key_0")

        data2 = io.BytesIO(b"second")
        item2 = self._create_byte_write_item(1)
        result2 = _call_write_item(stream, data2, item2, "key_1")

        self.assertEqual(result1.storage_data.offset, 0)
        self.assertEqual(result1.storage_data.length, 5)
        self.assertEqual(result2.storage_data.offset, 5)
        self.assertEqual(result2.storage_data.length, 6)

    def test_write_byte_io_large_data(self):
        """_write_item handles large BytesIO data."""
        stream = io.BytesIO()
        large_data = b"x" * 10000
        data = io.BytesIO(large_data)
        write_item = self._create_byte_write_item()
        storage_key = "large_key"

        result = _call_write_item(stream, data, write_item, storage_key)

        self.assertEqual(result.size_in_bytes, 10000)
        self.assertEqual(result.storage_data.length, 10000)

    def test_write_tensor_bool(self):
        """_write_item handles bool tensor."""
        stream = io.BytesIO()
        data = torch.tensor([True, False, True])
        write_item = self._create_tensor_write_item()
        storage_key = "bool_key"

        result = _call_write_item(stream, data, write_item, storage_key)
        self._assert_tensor_round_trip(stream, result, data)

        self.assertGreater(result.size_in_bytes, 0)

    def test_storage_info_attributes(self):
        """_StorageInfo stores correct relative_path, offset, length."""
        stream = io.BytesIO()
        data = io.BytesIO(b"test data")
        write_item = self._create_byte_write_item()
        storage_key = "verify_key"

        result = _call_write_item(stream, data, write_item, storage_key)

        sinfo = result.storage_data
        self.assertEqual(sinfo.relative_path, "verify_key")
        self.assertEqual(sinfo.offset, 0)
        self.assertEqual(sinfo.length, 9)

    def test_write_byte_io_preserves_stream_content(self):
        """_write_item writes correct content to stream."""
        stream = io.BytesIO()
        data = io.BytesIO(b"preserve me")
        write_item = self._create_byte_write_item()
        storage_key = "content_key"

        _call_write_item(stream, data, write_item, storage_key)

        self.assertEqual(stream.getvalue(), b"preserve me")

    def test_write_tensor_then_byte_io_sequential(self):
        """_write_item handles mixed type sequential writes."""
        stream = io.BytesIO()

        tensor_data = torch.tensor([1.0, 2.0])
        tensor_item = self._create_tensor_write_item(0)
        result1 = _call_write_item(stream, tensor_data, tensor_item, "tensor_key")
        self._assert_tensor_round_trip(stream, result1, tensor_data)

        byte_data = io.BytesIO(b"bytes")
        byte_item = self._create_byte_write_item(1)
        result2 = _call_write_item(stream, byte_data, byte_item, "byte_key")

        self.assertEqual(result1.storage_data.offset, 0)
        self.assertGreater(result1.storage_data.length, 0)
        self.assertEqual(result2.storage_data.offset, result1.storage_data.length)
        self.assertEqual(result2.storage_data.length, 5)

if __name__ == "__main__":
    run_tests()
