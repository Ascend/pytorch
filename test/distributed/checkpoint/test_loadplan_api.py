"""
1. PyTorch community lacks direct validation cases for some
   torch.distributed.checkpoint LoadPlan and LoadPlanner APIs, so this file is
   added.

2. This file validates the following APIs:
   torch.distributed.checkpoint.LoadPlan
   torch.distributed.checkpoint.LoadPlanner
   torch.distributed.checkpoint.LoadPlanner.set_up_planner
   torch.distributed.checkpoint.LoadPlanner.create_local_plan
   torch.distributed.checkpoint.LoadPlanner.create_global_plan
   torch.distributed.checkpoint.LoadPlanner.finish_plan
   torch.distributed.checkpoint.LoadPlanner.load_bytes
   torch.distributed.checkpoint.LoadPlanner.resolve_tensor
   torch.distributed.checkpoint.LoadPlanner.commit_tensor
   (extendable)
"""

import io
import tempfile

from torch_npu.testing.testcase import run_tests, TestCase

import torch
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    _create_default_local_metadata,
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan
from torch.distributed.checkpoint.planner_helpers import _create_read_item_for_tensor


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def _make_tensor_read_item(
    fqn="tensor",
    dest_offsets=(0, 0),
    lengths=(2, 2),
):
    zero_offsets = [0] * len(lengths)
    return _create_read_item_for_tensor(
        dest_index=MetadataIndex(fqn, zero_offsets),
        dest_offsets=dest_offsets,
        storage_index=MetadataIndex(fqn, zero_offsets),
        storage_offsets=zero_offsets,
        lengths=lengths,
    )


class MaterializeOnCpuLoadPlanner(DefaultLoadPlanner):
    """Planner that verifies commit_tensor can move loaded CPU data back to NPU."""

    def __init__(self):
        super().__init__()
        self.resolved_tensors = []
        self.committed_tensors = []

    def resolve_tensor(self, read_item):
        target = super().resolve_tensor(read_item)
        resolved = torch.empty_like(target, device="cpu")
        self.resolved_tensors.append(resolved)
        return resolved

    def commit_tensor(self, read_item, tensor):
        target = super().resolve_tensor(read_item)
        target.copy_(tensor.to(target.device))
        self.committed_tensors.append((read_item.dest_index.fqn, tensor.device.type))


class PlanDataLoadPlanner(DefaultLoadPlanner):
    """Planner that verifies local/global/finish plan customization."""

    def __init__(self):
        super().__init__()
        self.finished_storage_data = None
        self.finished_planner_data = None

    def create_local_plan(self):
        plan = super().create_local_plan()
        return LoadPlan(plan.items, planner_data={"local_plan": True})

    def create_global_plan(self, global_plan):
        return [
            LoadPlan(
                plan.items,
                storage_data={"storage_plan": index},
                planner_data={"global_plan": plan.planner_data},
            )
            for index, plan in enumerate(global_plan)
        ]

    def finish_plan(self, central_plan):
        self.finished_storage_data = central_plan.storage_data
        self.finished_planner_data = central_plan.planner_data
        return central_plan


class TestLoadPlanApi(TestCase):
    def test_default_load_planner_local_global_finish_plan(self):
        state_dict = {
            "tensor": torch.zeros(3, 4).to(device_type),
            "bytes": ["old"],
        }
        metadata_state_dict = {
            "tensor": torch.ones(3, 4),
            "bytes": ["new"],
        }
        metadata = _create_default_local_metadata(metadata_state_dict)

        planner = DefaultLoadPlanner()
        planner.set_up_planner(state_dict, metadata, is_coordinator=True)
        local_plan = planner.create_local_plan()

        self.assertIsInstance(local_plan, LoadPlan)
        self.assertEqual(2, len(local_plan.items))

        tensor_item = next(
            item for item in local_plan.items if item.dest_index.fqn == "tensor"
        )
        bytes_item = next(
            item for item in local_plan.items if item.dest_index.fqn == "bytes"
        )

        self.assertEqual(LoadItemType.TENSOR, tensor_item.type)
        self.assertEqual(torch.Size([0, 0]), tensor_item.dest_offsets)
        self.assertEqual(torch.Size([0, 0]), tensor_item.storage_offsets)
        self.assertEqual(torch.Size([3, 4]), tensor_item.lengths)
        self.assertEqual(LoadItemType.BYTE_IO, bytes_item.type)
        self.assertEqual(MetadataIndex("bytes"), bytes_item.dest_index)

        global_plan = planner.create_global_plan([local_plan])
        self.assertEqual([local_plan], global_plan)
        self.assertEqual(local_plan, planner.finish_plan(global_plan[0]))

    def test_default_load_planner_creates_multiple_tensor_read_items(self):
        state_dict = {"tensor": torch.zeros(8).to(device_type)}
        metadata = Metadata(
            state_dict_metadata={
                "tensor": TensorStorageMetadata(
                    properties=TensorProperties.create_from_tensor(torch.empty(8)),
                    size=torch.Size([8]),
                    chunks=[
                        ChunkStorageMetadata(
                            offsets=torch.Size([0]),
                            sizes=torch.Size([4]),
                        ),
                        ChunkStorageMetadata(
                            offsets=torch.Size([4]),
                            sizes=torch.Size([4]),
                        ),
                    ],
                ),
            },
        )

        planner = DefaultLoadPlanner()
        planner.set_up_planner(state_dict, metadata)
        local_plan = planner.create_local_plan()

        self.assertEqual(2, len(local_plan.items))
        low_item = next(
            item for item in local_plan.items if item.dest_offsets == torch.Size([0])
        )
        high_item = next(
            item for item in local_plan.items if item.dest_offsets == torch.Size([4])
        )

        self.assertEqual(LoadItemType.TENSOR, low_item.type)
        self.assertEqual(MetadataIndex("tensor", torch.Size([0])), low_item.dest_index)
        self.assertEqual(
            MetadataIndex("tensor", torch.Size([0])),
            low_item.storage_index,
        )
        self.assertEqual(torch.Size([0]), low_item.storage_offsets)
        self.assertEqual(torch.Size([4]), low_item.lengths)

        self.assertEqual(LoadItemType.TENSOR, high_item.type)
        self.assertEqual(
            MetadataIndex("tensor", torch.Size([0])),
            high_item.dest_index,
        )
        self.assertEqual(
            MetadataIndex("tensor", torch.Size([4])),
            high_item.storage_index,
        )
        self.assertEqual(torch.Size([0]), high_item.storage_offsets)
        self.assertEqual(torch.Size([4]), high_item.lengths)

    def test_default_load_planner_strict_and_partial_load(self):
        metadata = _create_default_local_metadata({"tensor": torch.ones(2, 2)})
        state_dict = {
            "tensor": torch.zeros(2, 2).to(device_type),
            "missing": torch.zeros(2, 2).to(device_type),
        }

        strict_planner = DefaultLoadPlanner(allow_partial_load=False)
        strict_planner.set_up_planner(state_dict, metadata)
        with self.assertRaisesRegex(RuntimeError, "Missing key in checkpoint"):
            strict_planner.create_local_plan()

        partial_planner = DefaultLoadPlanner(allow_partial_load=True)
        partial_planner.set_up_planner(state_dict, metadata)
        partial_plan = partial_planner.create_local_plan()
        self.assertEqual(1, len(partial_plan.items))
        self.assertEqual("tensor", partial_plan.items[0].dest_index.fqn)

    def test_default_load_planner_size_mismatch(self):
        metadata = _create_default_local_metadata({"tensor": torch.ones(2, 2)})
        state_dict = {"tensor": torch.zeros(3, 2).to(device_type)}

        planner = DefaultLoadPlanner()
        planner.set_up_planner(state_dict, metadata)
        with self.assertRaisesRegex(ValueError, "Size mismatch"):
            planner.create_local_plan()

    def test_resolve_tensor_returns_npu_narrow_view(self):
        state_dict = {"tensor": torch.zeros(4, 5).to(device_type)}
        metadata = _create_default_local_metadata({"tensor": torch.ones(4, 5)})
        read_item = _make_tensor_read_item(
            dest_offsets=[1, 2],
            lengths=[2, 2],
        )

        planner = DefaultLoadPlanner()
        planner.set_up_planner(state_dict, metadata)
        target_tensor = planner.resolve_tensor(read_item)

        self.assertEqual(device_type, target_tensor.device.type)
        self.assertEqual(torch.Size([2, 2]), target_tensor.size())

        target_tensor.copy_(torch.full((2, 2), 7.0))
        planner.commit_tensor(read_item, target_tensor)

        expected = torch.zeros(4, 5)
        expected[1:3, 2:4] = 7.0
        self.assertEqual(expected, state_dict["tensor"].cpu())

    def test_resolve_tensor_handles_non_contiguous_npu_target(self):
        npu_target = torch.zeros(5, 4).to(device_type).transpose(0, 1)
        self.assertFalse(npu_target.is_contiguous())
        state_dict = {"tensor": npu_target}
        metadata = _create_default_local_metadata({"tensor": torch.ones(4, 5)})
        read_item = _make_tensor_read_item(
            dest_offsets=[1, 1],
            lengths=[2, 3],
        )

        planner = DefaultLoadPlanner()
        planner.set_up_planner(state_dict, metadata)
        target_tensor = planner.resolve_tensor(read_item)

        self.assertEqual(device_type, target_tensor.device.type)
        self.assertEqual(torch.Size([2, 3]), target_tensor.size())
        target_tensor.copy_(torch.full((2, 3), 5.0))
        planner.commit_tensor(read_item, target_tensor)

        expected = torch.zeros(4, 5)
        expected[1:3, 1:4] = 5.0
        self.assertEqual(expected, state_dict["tensor"].cpu())

    def test_load_bytes_updates_flattened_original_state_dict(self):
        state_dict = {
            "nested": {
                "bytes": b"old",
            }
        }
        metadata = _create_default_local_metadata({"nested.bytes": b"new"})

        planner = DefaultLoadPlanner()
        planner.set_up_planner(state_dict, metadata)
        plan = planner.create_local_plan()
        read_item = next(
            item for item in plan.items if item.dest_index.fqn == "nested.bytes"
        )

        value = io.BytesIO()
        torch.save({"loaded": (1, 2, 3)}, value)
        value.seek(0)

        planner.load_bytes(read_item, value)
        self.assertEqual({"loaded": (1, 2, 3)}, state_dict["nested"]["bytes"])

    def test_load_bytes_updates_unflattened_state_dict(self):
        state_dict = {"payload": b"old"}
        metadata = _create_default_local_metadata({"payload": b"new"})

        planner = DefaultLoadPlanner(
            flatten_state_dict=False,
            flatten_sharded_tensors=False,
        )
        planner.set_up_planner(state_dict, metadata)
        plan = planner.create_local_plan()
        read_item = next(
            item for item in plan.items if item.dest_index.fqn == "payload"
        )

        value = io.BytesIO()
        torch.save({"loaded": (4, 5, 6)}, value)
        value.seek(0)

        planner.load_bytes(read_item, value)
        self.assertEqual({"loaded": (4, 5, 6)}, state_dict["payload"])


class TestLoadPlannerNpuIntegration(TestCase):
    def test_load_state_dict_accepts_custom_plan_data(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            state_dict_to_save = {
                "tensor": torch.arange(4, dtype=torch.float32)
                .reshape(2, 2)
                .to(device_type),
            }
            save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=FileSystemWriter(checkpoint_dir),
                planner=DefaultSavePlanner(),
                no_dist=True,
            )

            state_dict_to_load = {"tensor": torch.zeros(2, 2).to(device_type)}
            planner = PlanDataLoadPlanner()
            load_state_dict(
                state_dict=state_dict_to_load,
                storage_reader=FileSystemReader(checkpoint_dir),
                planner=planner,
                no_dist=True,
            )

            self.assertEqual(
                state_dict_to_save["tensor"].cpu(), state_dict_to_load["tensor"].cpu()
            )
            self.assertEqual({"storage_plan": 0}, planner.finished_storage_data)
            self.assertEqual(
                {"global_plan": {"local_plan": True}},
                planner.finished_planner_data,
            )

    def test_filesystem_metadata_version_when_supported(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            state_dict_to_save = {
                "tensor": torch.arange(4, dtype=torch.float32)
                .reshape(2, 2)
                .to(device_type),
            }
            save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=FileSystemWriter(checkpoint_dir),
                planner=DefaultSavePlanner(),
                no_dist=True,
            )

            metadata = FileSystemReader(checkpoint_dir).read_metadata()

            self.assertIsInstance(metadata, Metadata)
            if hasattr(metadata, "version"):
                from torch.distributed.checkpoint.filesystem import CURRENT_DCP_VERSION

                self.assertEqual(CURRENT_DCP_VERSION, metadata.version)

    def test_custom_commit_tensor_materializes_cpu_tensor_to_npu(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            state_dict_to_save = {
                "tensor": torch.arange(6, dtype=torch.float32)
                .reshape(2, 3)
                .to(device_type),
            }
            save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=FileSystemWriter(checkpoint_dir),
                planner=DefaultSavePlanner(),
                no_dist=True,
            )

            state_dict_to_load = {"tensor": torch.zeros(2, 3).to(device_type)}
            planner = MaterializeOnCpuLoadPlanner()
            load_state_dict(
                state_dict=state_dict_to_load,
                storage_reader=FileSystemReader(checkpoint_dir),
                planner=planner,
                no_dist=True,
            )

            self.assertEqual(
                state_dict_to_save["tensor"].cpu(), state_dict_to_load["tensor"].cpu()
            )
            self.assertEqual(1, len(planner.resolved_tensors))
            self.assertEqual("cpu", planner.resolved_tensors[0].device.type)
            self.assertEqual([("tensor", "cpu")], planner.committed_tensors)

    def test_filesystem_load_tensor_and_bytes_to_npu_state_dict(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            original_tensor = (
                torch.arange(20, dtype=torch.float32).reshape(4, 5).to(device_type)
            )
            state_dict_to_save = {
                "tensor": original_tensor,
                "payload": ["step", 3, "ok"],
            }
            save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=FileSystemWriter(checkpoint_dir),
                planner=DefaultSavePlanner(),
                no_dist=True,
            )

            state_dict_to_load = {
                "tensor": torch.full((4, 5), -1.0).to(device_type),
                "payload": [],
            }
            load_state_dict(
                state_dict=state_dict_to_load,
                storage_reader=FileSystemReader(checkpoint_dir),
                planner=DefaultLoadPlanner(),
                no_dist=True,
            )

            self.assertEqual(original_tensor.cpu(), state_dict_to_load["tensor"].cpu())
            self.assertEqual(["step", 3, "ok"], state_dict_to_load["payload"])


if __name__ == "__main__":
    run_tests()
