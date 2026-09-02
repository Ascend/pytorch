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
Add validation cases for torch.distributed.checkpoint APIs.

This file validates _EmptyStateDictLoadPlanner behavior.

Reasons:
1. PyTorch community lacks direct API validations for
   _EmptyStateDictLoadPlanner.
2. _EmptyStateDictLoadPlanner only handles checkpoint planning logic,
   which is device-independent.
3. These tests intentionally avoid NPU initialization and distributed
   environment requirements.

This file can be extended with more planner validation cases.
"""

import torch

from torch.distributed.checkpoint.default_planner import (
    _EmptyStateDictLoadPlanner,
)
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadPlan, LoadItemType
from torch.testing._internal.common_utils import TestCase, run_tests


def _make_tensor_md(size, dtype=torch.float32):
    """Helper to create TensorStorageMetadata for testing."""
    return TensorStorageMetadata(
        properties=TensorProperties.create_from_tensor(
            torch.empty(1, dtype=dtype)
        ),
        size=torch.Size(size),
        chunks=[
            ChunkStorageMetadata(
                offsets=torch.Size([0] * len(size)),
                sizes=torch.Size(size),
            )
        ],
    )


def _make_metadata(state_dict_metadata, *, planner_data=None):
    """Helper to create Metadata with a non-None planner_data default."""
    if planner_data is None:
        planner_data = {k: [k] for k in state_dict_metadata}
    return Metadata(
        state_dict_metadata=state_dict_metadata,
        planner_data=planner_data,
    )


class TestEmptyStateDictLoadPlanner(TestCase):

    # ---------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------

    def test_init_without_keys(self):
        """Verify planner initialization without specifying keys."""
        planner = _EmptyStateDictLoadPlanner()
        self.assertIsNone(planner.keys)

    def test_init_with_keys(self):
        """Verify planner initialization with specified keys."""
        keys = ["model"]
        planner = _EmptyStateDictLoadPlanner(keys=keys)
        self.assertEqual(planner.keys, keys)

    # ---------------------------------------------------------------
    # set_up_planner -- error paths
    # ---------------------------------------------------------------

    def test_set_up_planner_with_non_empty_state_dict(self):
        """_EmptyStateDictLoadPlanner requires an empty state_dict.

        A non-empty state_dict should trigger assertion failure.
        """
        planner = _EmptyStateDictLoadPlanner()
        with self.assertRaises(AssertionError):
            planner.set_up_planner(
                {"model": torch.tensor([1])},
                metadata=None,
            )

    def test_set_up_planner_without_metadata(self):
        """Metadata is required when setting up planner.

        Missing metadata should trigger assertion failure.
        """
        planner = _EmptyStateDictLoadPlanner()
        with self.assertRaises(AssertionError):
            planner.set_up_planner({}, metadata=None)

    # ---------------------------------------------------------------
    # set_up_planner -- success paths
    # ---------------------------------------------------------------

    def test_set_up_planner_success(self):
        """Empty state_dict with valid metadata: state_dict is populated.

        _EmptyStateDictLoadPlanner rebuilds the state_dict from metadata,
        creating empty tensors for each TensorStorageMetadata entry.
        After set_up_planner, the planner's internal state (metadata,
        state_dict) should be properly initialized.
        """
        planner = _EmptyStateDictLoadPlanner()
        metadata = _make_metadata(
            state_dict_metadata={
                "a": _make_tensor_md((2, 3)),
                "b": _make_tensor_md((4,), dtype=torch.float64),
            }
        )
        sd = {}
        planner.set_up_planner(sd, metadata)

        self.assertEqual(len(sd), 2)
        self.assertIn("a", sd)
        self.assertIn("b", sd)
        self.assertEqual(sd["a"].shape, torch.Size((2, 3)))
        self.assertEqual(sd["a"].dtype, torch.float32)
        self.assertEqual(sd["b"].shape, torch.Size((4,)))
        self.assertEqual(sd["b"].dtype, torch.float64)
        self.assertIs(planner.metadata, metadata)
        self.assertIsNotNone(planner.state_dict)

    def test_set_up_planner_with_is_coordinator(self):
        """set_up_planner should accept and store is_coordinator flag."""
        planner = _EmptyStateDictLoadPlanner()
        metadata = _make_metadata(
            state_dict_metadata={"a": _make_tensor_md((1,))}
        )
        sd = {}
        planner.set_up_planner(sd, metadata, is_coordinator=True)
        self.assertTrue(planner.is_coordinator)

    # ---------------------------------------------------------------
    # keys filtering in set_up_planner
    # ---------------------------------------------------------------

    def test_keys_none_loads_all(self):
        """When keys=None, every key in metadata is loaded."""
        planner = _EmptyStateDictLoadPlanner(keys=None)
        metadata = _make_metadata(
            state_dict_metadata={
                "x": _make_tensor_md((1,)),
                "y": _make_tensor_md((2,)),
                "z": _make_tensor_md((3,)),
            }
        )
        sd = {}
        planner.set_up_planner(sd, metadata)
        self.assertEqual(len(sd), 3)
        self.assertIn("x", sd)
        self.assertIn("y", sd)
        self.assertIn("z", sd)

    def test_keys_filter_loads_subset(self):
        """When keys is a specific set, only those keys are loaded."""
        planner = _EmptyStateDictLoadPlanner(keys={"x", "z"})
        metadata = _make_metadata(
            state_dict_metadata={
                "x": _make_tensor_md((1,)),
                "y": _make_tensor_md((2,)),
                "z": _make_tensor_md((3,)),
            },
            planner_data={"x": ["x"], "y": ["y"], "z": ["z"]},
        )
        sd = {}
        planner.set_up_planner(sd, metadata)
        self.assertEqual(len(sd), 2)
        self.assertIn("x", sd)
        self.assertIn("z", sd)
        self.assertNotIn("y", sd)

    def test_keys_filter_loads_nothing_when_no_match(self):
        """When keys matches nothing, state_dict stays empty."""
        planner = _EmptyStateDictLoadPlanner(keys={"nonexistent"})
        metadata = _make_metadata(
            state_dict_metadata={
                "a": _make_tensor_md((1,)),
                "b": _make_tensor_md((2,)),
            },
            planner_data={"a": ["a"], "b": ["b"]},
        )
        sd = {}
        planner.set_up_planner(sd, metadata)
        self.assertEqual(len(sd), 0)

    def test_keys_filter_with_planner_data(self):
        """keys filter works when metadata has planner_data (nested paths).

        The planner should match keys against both the storage key and
        the unflattened path components from planner_data.
        """
        planner = _EmptyStateDictLoadPlanner(keys={"model.layer.weight"})
        metadata = _make_metadata(
            state_dict_metadata={
                "0": _make_tensor_md((3, 4)),
            },
            planner_data={
                "0": ["model", "layer", "weight"],
            },
        )
        sd = {}
        planner.set_up_planner(sd, metadata)
        self.assertIn("model", sd)
        self.assertIn("layer", sd["model"])
        self.assertIn("weight", sd["model"]["layer"])
        self.assertEqual(
            sd["model"]["layer"]["weight"].shape, torch.Size((3, 4))
        )

    # ---------------------------------------------------------------
    # create_local_plan after set_up_planner
    # ---------------------------------------------------------------

    def test_create_local_plan_after_setup(self):
        """After successful set_up_planner, create_local_plan returns a
        LoadPlan with the expected number of ReadItems."""
        planner = _EmptyStateDictLoadPlanner()
        metadata = _make_metadata(
            state_dict_metadata={
                "a": _make_tensor_md((2, 3)),
                "b": _make_tensor_md((4,)),
            }
        )
        sd = {}
        planner.set_up_planner(sd, metadata)

        local_plan = planner.create_local_plan()
        self.assertIsInstance(local_plan, LoadPlan)
        self.assertGreaterEqual(len(local_plan.items), 2)
        item_types = {item.type for item in local_plan.items}
        self.assertIn(LoadItemType.TENSOR, item_types)

    def test_create_local_plan_empty_when_keys_filter_all(self):
        """When keys filter removes all metadata entries, local plan is empty."""
        planner = _EmptyStateDictLoadPlanner(keys={"nonexistent"})
        metadata = _make_metadata(
            state_dict_metadata={
                "a": _make_tensor_md((1,)),
            },
            planner_data={"a": ["a"]},
        )
        sd = {}
        planner.set_up_planner(sd, metadata)

        local_plan = planner.create_local_plan()
        self.assertIsInstance(local_plan, LoadPlan)
        self.assertEqual(len(local_plan.items), 0)

    # ---------------------------------------------------------------
    # create_global_plan
    # ---------------------------------------------------------------

    def test_create_global_plan(self):
        """create_global_plan should return a list of LoadPlan for each rank."""
        planner = _EmptyStateDictLoadPlanner()
        metadata = _make_metadata(
            state_dict_metadata={"a": _make_tensor_md((1,))}
        )
        sd = {}
        planner.set_up_planner(sd, metadata)
        local_plan = planner.create_local_plan()

        global_plans = planner.create_global_plan([local_plan])
        self.assertIsInstance(global_plans, list)
        self.assertEqual(len(global_plans), 1)
        self.assertIsInstance(global_plans[0], LoadPlan)

    def test_create_global_plan_multiple_ranks(self):
        """create_global_plan with plans from multiple ranks."""
        planner = _EmptyStateDictLoadPlanner()
        metadata = _make_metadata(
            state_dict_metadata={"a": _make_tensor_md((1,))}
        )
        sd = {}
        planner.set_up_planner(sd, metadata)
        p1 = planner.create_local_plan()
        p2 = planner.create_local_plan()

        global_plans = planner.create_global_plan([p1, p2])
        self.assertEqual(len(global_plans), 2)

    # ---------------------------------------------------------------
    # finish_plan
    # ---------------------------------------------------------------

    def test_finish_plan_passthrough(self):
        """finish_plan returns the plan unchanged (identity)."""
        planner = _EmptyStateDictLoadPlanner()
        metadata = _make_metadata(
            state_dict_metadata={"a": _make_tensor_md((1,))}
        )
        sd = {}
        planner.set_up_planner(sd, metadata)
        local_plan = planner.create_local_plan()

        finished_plan = planner.finish_plan(local_plan)
        self.assertIs(finished_plan, local_plan)


if __name__ == "__main__":
    run_tests()
