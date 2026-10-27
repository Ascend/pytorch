# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
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
Validate torch.distributed.checkpoint HuggingFace reader behavior on NPU.

The tests cover constructor input handling and read_data behavior with an
NPU destination tensor while isolating external storage dependencies.
"""

import sys
from unittest.mock import MagicMock, patch

import torch
from torch.distributed.checkpoint import _HuggingFaceStorageReader
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
)


class TestHuggingFaceStorageReader(TestCase):

    def test_read_data_hf(self) -> None:
        tensor_name = "weight"
        shard_name = "shard.safetensors"
        checkpoint_path = "hf://unit-test/repository"
        resolved_path = f"{checkpoint_path}/{shard_name}"

        source_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        destination = torch.empty_like(
            source_tensor,
            device=device_type,
        )

        read_item = MagicMock()
        read_item.storage_index.fqn = tensor_name
        read_item.dest_index.fqn = tensor_name

        load_plan = MagicMock()
        load_plan.items = [read_item]

        load_planner = MagicMock()
        load_planner.resolve_tensor.return_value = destination

        reader = _HuggingFaceStorageReader.__new__(
            _HuggingFaceStorageReader
        )
        reader.path = checkpoint_path
        reader.storage_data = {tensor_name: shard_name}
        reader.fs = MagicMock()
        reader.fs.concat_path.return_value = resolved_path

        stream = MagicMock()
        stream.read.return_value = b"mock-safetensors-data"
        reader.fs.create_stream.return_value.__enter__.return_value = stream

        safetensors_torch = MagicMock()
        safetensors_torch.load.return_value = {
            tensor_name: source_tensor,
        }

        with patch.dict(
            sys.modules,
            {
                "safetensors": MagicMock(),
                "safetensors.torch": safetensors_torch,
            },
        ):
            future = reader.read_data(
                load_plan,
                load_planner,
            )

        self.assertIsNone(future.wait())

        reader.fs.concat_path.assert_called_once_with(
            checkpoint_path,
            shard_name,
        )
        reader.fs.create_stream.assert_called_once_with(
            resolved_path,
            "rb",
        )
        safetensors_torch.load.assert_called_once_with(
            b"mock-safetensors-data"
        )
        load_planner.resolve_tensor.assert_called_once_with(read_item)
        self.assertEqual(load_planner.commit_tensor.call_count, 1)

        self.assertEqual(destination.device.type, device_type)
        self.assertEqual(destination.cpu(), source_tensor)

    def test_init_inputs_hf(self) -> None:
        mock_module = MagicMock()
        mock_module.HfFileSystem.protocol = "hf"

        with patch.dict(sys.modules, {"huggingface_hub": mock_module}):
            with patch(
                "torch.distributed.checkpoint._hf_storage.fsspec.available_protocols",
                return_value=["hf"],
            ):
                with patch.object(
                    FsspecReader,
                    "__init__",
                    return_value=None,
                ) as mock_init:
                    _HuggingFaceStorageReader(path="hf://test/repo")
                    mock_init.assert_called_once_with(
                        path="hf://test/repo",
                        token=None,
                    )

                with patch.object(
                    FsspecReader,
                    "__init__",
                    return_value=None,
                ) as mock_init:
                    _HuggingFaceStorageReader(
                        path="hf://test/repo",
                        token=None,
                    )
                    mock_init.assert_called_once_with(
                        path="hf://test/repo",
                        token=None,
                    )

                with patch.object(
                    FsspecReader,
                    "__init__",
                    return_value=None,
                ) as mock_init:
                    _HuggingFaceStorageReader(
                        path="hf://test/repo",
                        token="test_token",
                    )
                    mock_init.assert_called_once_with(
                        path="hf://test/repo",
                        token="test_token",
                    )

            with self.assertRaises(TypeError):
                _HuggingFaceStorageReader()


if __name__ == "__main__":
    run_tests()
