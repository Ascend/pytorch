# Owner(s): ["module: nn"]

"""
Add validation cases for torch.nn.utils.rnn.PackedSequence APIs on NPU:
1. PyTorch community lacks direct validations for PackedSequence.count,
   PackedSequence.index, and PackedSequence.is_pinned.
2. This file validates torch.nn.utils.rnn.PackedSequence,
   torch.nn.utils.rnn.PackedSequence.count,
   torch.nn.utils.rnn.PackedSequence.index, and
   torch.nn.utils.rnn.PackedSequence.is_pinned (extendable).
"""

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"


class TestPackedSequenceAPIs(TestCase):

    def test_packed_sequence_constructor_and_to_on_npu(self):
        data = torch.randn(5, 10, device=device_type)
        batch_sizes = torch.tensor([3, 2], dtype=torch.int64)
        sorted_indices = torch.tensor([2, 0, 1], dtype=torch.int64, device=device_type)
        unsorted_indices = torch.tensor(
            [1, 2, 0], dtype=torch.int64, device=device_type
        )

        packed = rnn_utils.PackedSequence(
            data, batch_sizes, sorted_indices, unsorted_indices
        )

        self.assertIsInstance(packed, rnn_utils.PackedSequence)
        self.assertEqual(packed.data.device.type, device_type)
        self.assertEqual(packed.data.shape, torch.Size([5, 10]))
        self.assertEqual(packed.batch_sizes.device.type, "cpu")
        self.assertEqual(packed.sorted_indices.device.type, device_type)
        self.assertEqual(packed.unsorted_indices.device.type, device_type)
        self.assertFalse(packed.is_pinned())

        packed_cpu = packed.to("cpu")
        self.assertEqual(packed_cpu.data.device.type, "cpu")
        self.assertEqual(packed_cpu.batch_sizes.device.type, "cpu")
        self.assertEqual(packed_cpu.sorted_indices.device.type, "cpu")
        self.assertEqual(packed_cpu.unsorted_indices.device.type, "cpu")

        packed_accelerator = packed_cpu.to(device_type)
        self.assertEqual(packed_accelerator.data.device.type, device_type)
        self.assertEqual(packed_accelerator.batch_sizes.device.type, "cpu")
        self.assertEqual(packed_accelerator.sorted_indices.device.type, device_type)
        self.assertEqual(packed_accelerator.unsorted_indices.device.type, device_type)

    def test_packed_sequence_namedtuple_methods_on_optional_indices(self):
        data = torch.tensor([1.0, 2.0, 3.0], device=device_type)
        batch_sizes = torch.tensor([2, 1], dtype=torch.int64)
        packed = rnn_utils.PackedSequence(data, batch_sizes)

        self.assertIsNone(packed.sorted_indices)
        self.assertIsNone(packed.unsorted_indices)
        self.assertEqual(packed.count(None), 2)
        self.assertEqual(packed.index(None), 2)
        self.assertEqual(packed.count("non_existent"), 0)
        with self.assertRaises(ValueError):
            packed.index("non_existent")

    def test_packed_sequence_rejects_accelerator_batch_sizes(self):
        data = torch.tensor([1.0, 2.0], device=device_type)
        batch_sizes = torch.tensor([2], dtype=torch.int64, device=device_type)

        if device_type == "cpu":
            packed = rnn_utils.PackedSequence(data, batch_sizes)
            self.assertEqual(packed.batch_sizes.device.type, "cpu")
        else:
            with self.assertRaisesRegex(
                ValueError, "batch_sizes should always be on CPU"
            ):
                rnn_utils.PackedSequence(data, batch_sizes)


if __name__ == "__main__":
    run_tests()
