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
Tests for deprecated distributed checkpoint state-dict load/save APIs.

These tests validate the core save and load behavior with default and explicit
planners, distributed and non-distributed execution, representative state-dict
value types, optional arguments, and failure boundaries. The coverage focuses
on public API behavior so additional storage backends or planners can reuse the
same validation without depending on internal implementation details.
"""

import os
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import (
    CheckpointException,
    FileSystemReader,
    FileSystemWriter,
    Metadata,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from torch.distributed.checkpoint.state_dict_saver import save_state_dict
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def _create_source_state_dict():
    return {
        "tensor": torch.arange(
            6, dtype=torch.float32
        ).reshape(2, 3).to(device_type),
        "empty": torch.empty(
            0, dtype=torch.float32
        ).to(device_type),
        "nested": {
            "tensor": torch.tensor(
                [1, 2], dtype=torch.int64
            ).to(device_type),
        },
        "value": 1,
    }


def _create_target_state_dict():
    return {
        "tensor": torch.empty(
            2, 3, dtype=torch.float32
        ).to(device_type),
        "empty": torch.empty(
            0, dtype=torch.float32
        ).to(device_type),
        "nested": {
            "tensor": torch.empty(
                2, dtype=torch.int64
            ).to(device_type),
        },
        "value": 0,
    }


class StateDictLoaderSaverTest(TestCase):
    def test_no_dist_round_trip(self):
        source_state_dict = _create_source_state_dict()

        with TemporaryDirectory() as checkpoint_dir:
            metadata = save_state_dict(
                source_state_dict,
                FileSystemWriter(checkpoint_dir),
                no_dist=True,
            )

            self.assertIsInstance(metadata, Metadata)

            target_state_dict = _create_target_state_dict()
            result = load_state_dict(
                target_state_dict,
                FileSystemReader(checkpoint_dir),
                no_dist=True,
            )

            self.assertIsNone(result)
            self.assertEqual(target_state_dict, source_state_dict)
            self.assertEqual(
                target_state_dict["tensor"].device.type,
                device_type,
            )

    def test_no_dist_round_trip_with_planners(self):
        source_state_dict = _create_source_state_dict()

        with TemporaryDirectory() as checkpoint_dir:
            metadata = save_state_dict(
                state_dict=source_state_dict,
                storage_writer=FileSystemWriter(checkpoint_dir),
                process_group=None,
                coordinator_rank=0,
                no_dist=True,
                planner=DefaultSavePlanner(),
            )

            self.assertIsInstance(metadata, Metadata)

            target_state_dict = _create_target_state_dict()
            result = load_state_dict(
                state_dict=target_state_dict,
                storage_reader=FileSystemReader(checkpoint_dir),
                process_group=None,
                coordinator_rank=0,
                no_dist=True,
                planner=DefaultLoadPlanner(),
            )

            self.assertIsNone(result)
            self.assertEqual(target_state_dict, source_state_dict)

    def test_empty_state_dict(self):
        with TemporaryDirectory() as checkpoint_dir:
            metadata = save_state_dict(
                {},
                FileSystemWriter(checkpoint_dir),
                no_dist=True,
            )

            self.assertIsInstance(metadata, Metadata)
            self.assertEqual(metadata.state_dict_metadata, {})

            state_dict = {}
            result = load_state_dict(
                state_dict,
                FileSystemReader(checkpoint_dir),
                no_dist=True,
            )

            self.assertIsNone(result)
            self.assertEqual(state_dict, {})

    def test_load_missing_key_raises(self):
        source_state_dict = {
            "tensor": torch.ones(
                2, dtype=torch.float32
            ).to(device_type),
        }

        with TemporaryDirectory() as checkpoint_dir:
            save_state_dict(
                source_state_dict,
                FileSystemWriter(checkpoint_dir),
                no_dist=True,
            )

            target_state_dict = {
                "missing": torch.zeros(
                    2, dtype=torch.float32
                ).to(device_type),
            }

            with self.assertRaises(CheckpointException):
                load_state_dict(
                    target_state_dict,
                    FileSystemReader(checkpoint_dir),
                    no_dist=True,
                )


class StateDictLoaderSaverDistributedTest(TestCase):
    def setUp(self):
        super().setUp()

        self.temp_dir = TemporaryDirectory()
        self.store = dist.FileStore(
            os.path.join(self.temp_dir.name, "store"),
            1,
        )
        dist.init_process_group(
            backend="gloo",
            store=self.store,
            rank=0,
            world_size=1,
        )

    def tearDown(self):
        dist.destroy_process_group()
        self.temp_dir.cleanup()

        super().tearDown()

    def test_distributed_round_trip(self):
        checkpoint_dir = os.path.join(
            self.temp_dir.name,
            "checkpoint",
        )
        source_state_dict = {
            "tensor": torch.tensor(
                [1.0, 2.0]
            ).to(device_type),
        }

        metadata = save_state_dict(
            state_dict=source_state_dict,
            storage_writer=FileSystemWriter(checkpoint_dir),
            process_group=dist.group.WORLD,
            coordinator_rank=0,
            no_dist=False,
        )

        self.assertIsInstance(metadata, Metadata)

        target_state_dict = {
            "tensor": torch.zeros(
                2
            ).to(device_type),
        }

        result = load_state_dict(
            state_dict=target_state_dict,
            storage_reader=FileSystemReader(checkpoint_dir),
            process_group=dist.group.WORLD,
            coordinator_rank=0,
            no_dist=False,
        )

        self.assertIsNone(result)
        self.assertEqual(target_state_dict, source_state_dict)


if __name__ == "__main__":
    run_tests()
