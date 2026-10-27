# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Licensed under the BSD 3-Clause License.

import os
import unittest

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


@instantiate_parametrized_tests
@unittest.skipIf(not torch_npu.npu.is_available(), "NPU is required")
class TestMemoryHistoryBuffer(TestCase):
    def setUp(self):
        super().setUp()
        torch.empty(1, device="npu")
        torch_npu.npu.synchronize()
        torch_npu.npu.memory._record_memory_history(None)
        torch_npu.npu.empty_cache()
        self.device = torch_npu.npu.current_device()

    def tearDown(self):
        torch_npu.npu.memory._record_memory_history(None)
        torch_npu.npu.synchronize()
        torch_npu.npu.empty_cache()
        super().tearDown()

    def test_memory_history_buffer(self):
        memory = torch_npu.npu.memory
        memory._record_memory_history(context=None, stacks="python", max_entries=3)
        tensors = [torch.empty(4096, dtype=torch.uint8, device="npu") for _ in range(8)]
        torch_npu.npu.synchronize()
        traces = memory._snapshot()["device_traces"][self.device]
        self.assertEqual([event["action"] for event in traces], ["alloc"] * 3)
        self.assertEqual([event["addr"] for event in traces], [tensor.data_ptr() for tensor in tensors[-3:]])

        memory._record_memory_history(None)
        self.assertEqual(memory._snapshot()["device_traces"][self.device], [])
        memory._record_memory_history(context=None, stacks="python", max_entries=3)
        tensor = torch.empty(4096, dtype=torch.uint8, device="npu")
        torch_npu.npu.synchronize()
        traces = memory._snapshot()["device_traces"][self.device]
        self.assertEqual([event["action"] for event in traces], ["alloc"])
        self.assertEqual(traces[0]["addr"], tensor.data_ptr())

    def test_memory_history_free_events(self):
        memory = torch_npu.npu.memory
        memory._record_memory_history(context=None, stacks="python", max_entries=16)
        tensor = torch.empty(4096, dtype=torch.uint8, device="npu")
        address = tensor.data_ptr()
        del tensor
        torch_npu.npu.synchronize()
        torch_npu.npu.empty_cache()
        traces = memory._snapshot()["device_traces"][self.device]
        actions = [event["action"] for event in traces if event["addr"] == address
                   and event["action"] in ("alloc", "free_requested", "free_completed")]
        self.assertEqual(actions, ["alloc", "free_requested", "free_completed"])
        self.assertTrue(any(event["action"] in ("segment_free", "segment_unmap") for event in traces))

    def test_memory_history_single_entry(self):
        memory = torch_npu.npu.memory
        memory._record_memory_history(context=None, stacks="python", max_entries=1)
        tensor = torch.empty(4096, dtype=torch.uint8, device="npu")
        torch_npu.npu.synchronize()
        traces = memory._snapshot()["device_traces"][self.device]
        self.assertEqual([event["action"] for event in traces], ["alloc"])
        self.assertEqual(traces[0]["addr"], tensor.data_ptr())
        self.assertEqual([event["action"] for event in memory._snapshot()["device_traces"][self.device]], ["snapshot"])

    def test_memory_history_grow_capacity(self):
        memory = torch_npu.npu.memory
        memory._record_memory_history(context=None, stacks="python", max_entries=3)
        tensors = [torch.empty(4096, dtype=torch.uint8, device="npu") for _ in range(8)]
        torch_npu.npu.synchronize()
        self.assertEqual(len(memory._snapshot()["device_traces"][self.device]), 3)
        memory._record_memory_history(context=None, stacks="python", max_entries=6)
        tensors.extend(torch.empty(4096, dtype=torch.uint8, device="npu") for _ in range(8))
        torch_npu.npu.synchronize()
        traces = memory._snapshot()["device_traces"][self.device]
        self.assertEqual([event["action"] for event in traces], ["alloc"] * 6)
        self.assertEqual([event["addr"] for event in traces], [tensor.data_ptr() for tensor in tensors[-6:]])

    @parametrize("skip_actions", [
        None, [], ["alloc"], ["alloc", "alloc"], ["free_requested"],
        ["free_requested", "free_completed"], ["snapshot"],
        ["segment_alloc", "segment_free", "segment_map", "segment_unmap"],
        ["alloc", "free_requested", "free_completed", "segment_alloc", "segment_free",
         "segment_map", "segment_unmap", "snapshot", "oom", "workspace_snapshot"],
    ])
    def test_skip_actions(self, skip_actions):
        memory = torch_npu.npu.memory
        for skipped in (["alloc"], skip_actions, None):
            memory._record_memory_history(context=None, stacks="python", skip_actions=skipped)
            baseline = memory.memory_allocated()
            tensor = torch.empty(4 * 1024 * 1024, dtype=torch.uint8, device="npu")
            address = tensor.data_ptr()
            torch_npu.npu.synchronize()
            snapshot = memory._snapshot()
            self.assertGreater(memory.memory_allocated(), baseline)
            self.assertTrue(any(
                segment["address"] <= address < segment["address"] + segment["total_size"]
                for segment in snapshot["segments"] if segment["device"] == self.device
            ))
            del tensor
            torch_npu.npu.synchronize()
            self.assertEqual(memory.memory_allocated(), baseline)
            torch_npu.npu.empty_cache()
            actions = {event["action"] for event in memory._snapshot()["device_traces"][self.device]}
            self.assertFalse(actions.intersection(skipped or []))
            expected = {"alloc", "free_requested", "free_completed", "snapshot"} - set(skipped or [])
            self.assertTrue(expected.issubset(actions))

    @parametrize("skip_actions,error", [
        (["alloc", "invalid"], RuntimeError), (["annotate"], RuntimeError),
        ("alloc", TypeError), ([1], TypeError),
    ])
    def test_skip_actions_invalid(self, skip_actions, error):
        memory = torch_npu.npu.memory
        memory._record_memory_history(context=None, stacks="python", skip_actions=["snapshot"])
        tensor = torch.empty(4096, device="npu")
        torch_npu.npu.synchronize()
        before = memory._snapshot()["device_traces"][self.device]
        self.assertTrue(any(event["addr"] == tensor.data_ptr() for event in before))
        with self.assertRaises(error):
            memory._record_memory_history(skip_actions=skip_actions)
        self.assertTrue(torch_npu._C._npu_isHistoryEnabled())
        self.assertEqual(memory._snapshot()["device_traces"][self.device], before)

    @parametrize("skip_actions", [
        None, [], ["workspace_snapshot"], ["segment_alloc"], ["alloc"],
        ["workspace_snapshot", "segment_alloc", "alloc"],
    ])
    def test_skip_actions_workspace(self, skip_actions):
        from torch.utils.cpp_extension import load_inline

        npu_lib = os.path.join(os.path.dirname(torch_npu.__file__), "lib")
        extension = load_inline(
            name="snapshot_workspace_extension",
            cpp_sources="""
                namespace at_npu::native { at::Tensor allocate_workspace(uint64_t, void*); }
                at::Tensor allocate_snapshot_workspace(uint64_t stream) {
                    return at_npu::native::allocate_workspace(4096, reinterpret_cast<void*>(stream));
                }
            """,
            functions=["allocate_snapshot_workspace"],
            extra_ldflags=[f"-L{npu_lib}", "-ltorch_npu", f"-Wl,-rpath,{npu_lib}"],
        )
        memory = torch_npu.npu.memory
        memory._record_memory_history(context=None, stacks="python")
        workspace = extension.allocate_snapshot_workspace(torch_npu.npu.current_stream().npu_stream)
        address = workspace.untyped_storage().data_ptr()
        baseline = memory._snapshot()
        expected = {"workspace_snapshot", "segment_alloc", "alloc"}
        self.assertEqual({event["action"] for event in baseline["device_traces"][self.device]
                          if event["addr"] == address}, expected)
        segments = [segment for segment in baseline["segments"] if segment["address"] == address]
        self.assertEqual(len(segments), 1)
        for skipped in (skip_actions, None):
            memory._record_memory_history(context=None, stacks="python", skip_actions=skipped)
            snapshot = memory._snapshot()
            self.assertEqual({event["action"] for event in snapshot["device_traces"][self.device]
                              if event["addr"] == address}, expected - set(skipped or []))
            self.assertEqual([segment for segment in snapshot["segments"] if segment["address"] == address], segments)


if __name__ == "__main__":
    run_tests()
