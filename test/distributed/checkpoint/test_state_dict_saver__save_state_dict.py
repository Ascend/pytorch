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
Add focused NPU validation cases for
torch.distributed.checkpoint.state_dict_saver._save_state_dict.

This file validates _save_state_dict on NPU in single-process no_dist mode.
It focuses on the _save_state_dict API itself: state_dict, storage_writer,
planner, and use_collectives.
"""

import inspect
import os
import tempfile

import torch
from torch.distributed.checkpoint import DefaultSavePlanner, FileSystemWriter
from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
from torch.testing._internal.common_utils import TestCase, run_tests


_SAVE_STATE_DICT_SIGNATURE = inspect.signature(_save_state_dict)


class RecordingSavePlanner(DefaultSavePlanner):
    """Record SavePlanner calls to verify _save_state_dict planner flow."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.is_coordinator = None

    def set_up_planner(
        self,
        state_dict,
        storage_meta=None,
        is_coordinator=False,
    ):
        self.calls.append("set_up_planner")
        self.is_coordinator = is_coordinator

        super_signature = inspect.signature(super().set_up_planner)

        if "storage_meta" in super_signature.parameters:
            return super().set_up_planner(
                state_dict,
                storage_meta=storage_meta,
                is_coordinator=is_coordinator,
            )

        return super().set_up_planner(
            state_dict,
            is_coordinator=is_coordinator,
        )

    def create_local_plan(self):
        self.calls.append("create_local_plan")
        return super().create_local_plan()

    def create_global_plan(self, all_plans):
        self.calls.append("create_global_plan")
        return super().create_global_plan(all_plans)

    def finish_plan(self, new_plan):
        self.calls.append("finish_plan")
        return super().finish_plan(new_plan)


class TestSaveStateDictNpu(TestCase):
    """Single-process NPU tests for _save_state_dict."""

    def _save_state_dict(
        self,
        state_dict,
        checkpoint_dir,
        planner=None,
        use_collectives=True,
    ):
        kwargs = {
            "state_dict": state_dict,
            "storage_writer": FileSystemWriter(checkpoint_dir),
            "process_group": None,
            "coordinator_rank": 0,
            "no_dist": True,
            "planner": planner,
        }

        if "use_collectives" in _SAVE_STATE_DICT_SIGNATURE.parameters:
            kwargs["use_collectives"] = use_collectives

        torch.accelerator.synchronize()
        metadata = _save_state_dict(**kwargs)
        torch.accelerator.synchronize()

        return metadata

    def _assert_checkpoint_written(self, checkpoint_dir):
        checkpoint_files = os.listdir(checkpoint_dir)

        metadata_files = [
            filename
            for filename in checkpoint_files
            if filename.endswith(".metadata")
        ]
        distcp_files = [
            filename
            for filename in checkpoint_files
            if filename.endswith(".distcp")
        ]

        self.assertGreater(
            len(metadata_files),
            0,
            f"No metadata file found in {checkpoint_files}",
        )
        self.assertGreater(
            len(distcp_files),
            0,
            f"No distcp file found in {checkpoint_files}",
        )

    def test_save_npu_tensor_no_dist(self):
        device = torch.accelerator.current_accelerator()
        if device is None or device.type != "npu":
            self.skipTest("NPU is not available in this test environment.")
        if torch.accelerator.device_count() == 0:
            self.skipTest("NPU device count is 0 in this test environment.")

        with tempfile.TemporaryDirectory(
            prefix=f"{self.__class__.__name__}_"
        ) as checkpoint_dir:
            state_dict = {
                "tensor": torch.arange(
                    12,
                    dtype=torch.float32,
                    device=device,
                ).reshape(3, 4),
            }

            metadata = self._save_state_dict(
                state_dict=state_dict,
                checkpoint_dir=checkpoint_dir,
            )

            self.assertIsNotNone(metadata)
            self.assertIn(
                "tensor",
                metadata.state_dict_metadata,
            )
            self._assert_checkpoint_written(checkpoint_dir)

    def test_save_nested_npu_state_dict_no_dist(self):
        device = torch.accelerator.current_accelerator()
        if device is None or device.type != "npu":
            self.skipTest("NPU is not available in this test environment.")
        if torch.accelerator.device_count() == 0:
            self.skipTest("NPU device count is 0 in this test environment.")

        with tempfile.TemporaryDirectory(
            prefix=f"{self.__class__.__name__}_"
        ) as checkpoint_dir:
            state_dict = {
                "layer": {
                    "weight": torch.ones(
                        2,
                        3,
                        dtype=torch.float32,
                        device=device,
                    ),
                    "bias": torch.arange(
                        3,
                        dtype=torch.float32,
                        device=device,
                    ),
                }
            }

            metadata = self._save_state_dict(
                state_dict=state_dict,
                checkpoint_dir=checkpoint_dir,
            )

            self.assertIsNotNone(metadata)
            self.assertIn(
                "layer.weight",
                metadata.state_dict_metadata,
            )
            self.assertIn(
                "layer.bias",
                metadata.state_dict_metadata,
            )
            self._assert_checkpoint_written(checkpoint_dir)

    def test_save_npu_tensor_with_custom_planner_no_dist(self):
        device = torch.accelerator.current_accelerator()
        if device is None or device.type != "npu":
            self.skipTest("NPU is not available in this test environment.")
        if torch.accelerator.device_count() == 0:
            self.skipTest("NPU device count is 0 in this test environment.")

        with tempfile.TemporaryDirectory(
            prefix=f"{self.__class__.__name__}_"
        ) as checkpoint_dir:
            state_dict = {
                "tensor": torch.arange(
                    6,
                    dtype=torch.float32,
                    device=device,
                ),
            }
            planner = RecordingSavePlanner()

            metadata = self._save_state_dict(
                state_dict=state_dict,
                checkpoint_dir=checkpoint_dir,
                planner=planner,
            )

            self.assertIsNotNone(metadata)
            self.assertIn(
                "tensor",
                metadata.state_dict_metadata,
            )
            self._assert_checkpoint_written(checkpoint_dir)

            self.assertTrue(planner.is_coordinator)
            self.assertEqual(
                planner.calls,
                [
                    "set_up_planner",
                    "create_local_plan",
                    "create_global_plan",
                    "finish_plan",
                ],
            )

    def test_save_npu_tensor_with_collectives_disabled_no_dist(self):
        if "use_collectives" not in _SAVE_STATE_DICT_SIGNATURE.parameters:
            self.skipTest(
                "use_collectives not supported in _save_state_dict."
            )

        device = torch.accelerator.current_accelerator()
        if device is None or device.type != "npu":
            self.skipTest("NPU is not available in this test environment.")
        if torch.accelerator.device_count() == 0:
            self.skipTest("NPU device count is 0 in this test environment.")

        with tempfile.TemporaryDirectory(
            prefix=f"{self.__class__.__name__}_"
        ) as checkpoint_dir:
            state_dict = {
                "tensor": torch.arange(
                    5,
                    dtype=torch.float32,
                    device=device,
                ),
            }

            metadata = self._save_state_dict(
                state_dict=state_dict,
                checkpoint_dir=checkpoint_dir,
                use_collectives=False,
            )

            self.assertIsNotNone(metadata)
            self.assertIn(
                "tensor",
                metadata.state_dict_metadata,
            )
            self._assert_checkpoint_written(checkpoint_dir)


if __name__ == "__main__":
    run_tests()
