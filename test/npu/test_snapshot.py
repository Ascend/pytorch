# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Licensed under the BSD 3-Clause License.

import unittest

import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


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


if __name__ == "__main__":
    run_tests()
