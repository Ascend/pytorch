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
Tests for DynamicMetaLoadPlanner setup behavior.

These tests validate dynamic metadata generation, optional metadata handling,
coordinator propagation, state-dict flattening, representative tensor types,
and supported input boundaries. The coverage focuses on public planner
behavior so future metadata implementations can evolve without requiring
tests to depend on unnecessary internal details.
"""

import torch
from torch.distributed.checkpoint.format_utils import DynamicMetaLoadPlanner
from torch.distributed.checkpoint.metadata import Metadata
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class DynamicMetaLoadPlannerTest(TestCase):
    def test_set_up_planner_default(self):
        state_dict = {
            "float_tensor": torch.arange(
                6, dtype=torch.float32
            ).reshape(2, 3).to(device_type),
            "empty_tensor": torch.empty(
                0, dtype=torch.float32
            ).to(device_type),
            "nested": {
                "tensor": torch.tensor(
                    [1, 2], dtype=torch.int64
                ).to(device_type),
            },
        }

        planner = DynamicMetaLoadPlanner()
        result = planner.set_up_planner(state_dict)

        self.assertIsNone(result)
        self.assertFalse(planner.is_coordinator)
        self.assertIsInstance(planner.metadata, Metadata)

        self.assertEqual(
            set(planner.metadata.state_dict_metadata.keys()),
            {
                "float_tensor",
                "empty_tensor",
                "nested.tensor",
            },
        )

        self.assertEqual(
            planner.metadata.state_dict_metadata[
                "float_tensor"
            ].properties.dtype,
            torch.float32,
        )
        self.assertEqual(
            planner.metadata.state_dict_metadata[
                "float_tensor"
            ].size,
            torch.Size([2, 3]),
        )
        self.assertEqual(
            planner.metadata.state_dict_metadata[
                "nested.tensor"
            ].properties.dtype,
            torch.int64,
        )
        self.assertEqual(
            planner.metadata.state_dict_metadata[
                "empty_tensor"
            ].size,
            torch.Size([0]),
        )

    def test_set_up_planner_with_explicit_options(self):
        state_dict = {
            "tensor": torch.tensor(
                [1.0, 2.0], dtype=torch.float32
            ).to(device_type),
        }
        metadata = Metadata(state_dict_metadata={})

        planner = DynamicMetaLoadPlanner(
            flatten_state_dict=False,
            flatten_sharded_tensors=False,
            allow_partial_load=True,
        )

        result = planner.set_up_planner(
            state_dict=state_dict,
            metadata=metadata,
            is_coordinator=True,
        )

        self.assertIsNone(result)
        self.assertTrue(planner.is_coordinator)
        self.assertIsNot(planner.metadata, metadata)
        self.assertEqual(
            set(planner.metadata.state_dict_metadata.keys()),
            {"tensor"},
        )
        self.assertEqual(
            planner.metadata.state_dict_metadata[
                "tensor"
            ].properties.dtype,
            torch.float32,
        )

    def test_set_up_planner_empty_state_dict(self):
        planner = DynamicMetaLoadPlanner()

        result = planner.set_up_planner({})

        self.assertIsNone(result)
        self.assertIsInstance(planner.metadata, Metadata)
        self.assertEqual(
            planner.metadata.state_dict_metadata,
            {},
        )

    def test_set_up_planner_rejects_non_tensor_value(self):
        planner = DynamicMetaLoadPlanner()

        with self.assertRaisesRegex(
            RuntimeError,
            "Non-tensor value identified",
        ):
            planner.set_up_planner(
                {
                    "tensor": torch.ones(
                        2, dtype=torch.float32
                    ).to(device_type),
                    "value": 1,
                }
            )


if __name__ == "__main__":
    run_tests()
