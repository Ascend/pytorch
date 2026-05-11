"""
1. PyTorch community lacks direct validation cases for some
   torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader APIs,
   so this file is added.

2. This file validates the following APIs:
   torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_metadata
   torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.prepare_local_plan
   torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.prepare_global_plan
   torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_data
   (extendable)
"""

import os
import tempfile

from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU, with_comms
from torch_npu.testing.testcase import run_tests, TestCase

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.format_utils import (
    BroadcastingTorchSaveReader,
    DynamicMetaLoadPlanner,
)
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, ReadItem
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def _make_byteio_read_item(fqn: str = "payload") -> ReadItem:
    return ReadItem(
        type=LoadItemType.BYTE_IO,
        dest_index=MetadataIndex(fqn),
        dest_offsets=torch.Size((0,)),
        storage_index=MetadataIndex(fqn),
        storage_offsets=torch.Size((0,)),
        lengths=torch.Size((0,)),
    )


class DummyLoadPlanner:
    flatten_state_dict = False
    state_dict = {}


class TestBroadcastingTorchSaveReaderApi(TestCase):
    def test_read_metadata_returns_empty_metadata(self):
        reader = BroadcastingTorchSaveReader(checkpoint_id="unused.pt")

        metadata = reader.read_metadata()

        self.assertIsInstance(metadata, Metadata)
        self.assertEqual({}, metadata.state_dict_metadata)

    def test_prepare_local_plan_returns_input_plan(self):
        reader = BroadcastingTorchSaveReader()
        plan = LoadPlan([], storage_data={"storage": 1}, planner_data={"planner": 2})

        result = reader.prepare_local_plan(plan)

        self.assertIs(result, plan)

    def test_prepare_global_plan_returns_input_plans(self):
        reader = BroadcastingTorchSaveReader()
        plans = [
            LoadPlan([], storage_data={"rank": 0}),
            LoadPlan([], storage_data={"rank": 1}),
        ]

        result = reader.prepare_global_plan(plans)

        self.assertIs(result, plans)

    def test_read_data_rejects_byte_io_items(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            torch_path = os.path.join(temp_dir, "payload.pt")
            torch.save({"payload": ["step", 1]}, torch_path)

            reader = BroadcastingTorchSaveReader(checkpoint_id=torch_path)
            reader.is_coordinator = True
            plan = LoadPlan([_make_byteio_read_item("payload")])

            with self.assertRaisesRegex(RuntimeError, "only supports loading Tensors"):
                reader.read_data(plan, DummyLoadPlanner())


class TestBroadcastingTorchSaveReaderNpu(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    @with_temp_dir
    @skipIfUnsupportMultiNPU(2)
    def test_read_data_loads_torch_save_tensor_to_npu_state_dict(self):
        source_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        torch_path = os.path.join(self.temp_dir, "tensor.pt")
        if dist.get_rank() == 0:
            torch.save({"tensor": source_tensor}, torch_path)
        dist.barrier()

        state_dict = {"tensor": torch.zeros(3, 4).to(device_type)}
        dcp.load(
            state_dict,
            planner=DynamicMetaLoadPlanner(),
            storage_reader=BroadcastingTorchSaveReader(),
            checkpoint_id=torch_path,
        )

        self.assertEqual(source_tensor, state_dict["tensor"].cpu())

    @with_comms
    @with_temp_dir
    @skipIfUnsupportMultiNPU(2)
    def test_read_data_handles_nested_state_dict(self):
        source_state_dict = {
            "model": {
                "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                "bias": torch.tensor([1.0, 2.0], dtype=torch.float32),
            }
        }
        torch_path = os.path.join(self.temp_dir, "nested.pt")
        if dist.get_rank() == 0:
            torch.save(source_state_dict, torch_path)
        dist.barrier()

        state_dict = {
            "model": {
                "weight": torch.zeros(2, 3).to(device_type),
                "bias": torch.zeros(2).to(device_type),
            }
        }
        dcp.load(
            state_dict,
            planner=DynamicMetaLoadPlanner(),
            storage_reader=BroadcastingTorchSaveReader(),
            checkpoint_id=torch_path,
        )

        self.assertEqual(
            source_state_dict["model"]["weight"], state_dict["model"]["weight"].cpu()
        )
        self.assertEqual(
            source_state_dict["model"]["bias"], state_dict["model"]["bias"].cpu()
        )


if __name__ == "__main__":
    run_tests()
