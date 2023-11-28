import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestPadPackedSequence(TestCase):
    @unittest.skip("skip test_pad_packed_sequence_fp32 now")
    def test_pad_packed_sequence_fp32(self, device="npu"):
        data = torch.tensor([4, 1, 3, 5, 2, 6], dtype=torch.float32)
        batch_sizes = torch.tensor([3, 2, 1], dtype=torch.int64)
        cpu_out, cpu_lengths = torch._pad_packed_sequence(data, batch_sizes, False, 0, 6)
        npu_out, npu_lengths = torch._pad_packed_sequence(data.npu(), batch_sizes, False, 0, 6)
        self.assertRtolEqual(cpu_out, npu_out.cpu())
        self.assertRtolEqual(cpu_lengths, npu_lengths.cpu())

    @unittest.skip("skip test_pad_packed_sequence_fp16 now")
    def test_pad_packed_sequence_fp16(self, device="npu"):
        data = torch.tensor([4, 1, 3, 5, 2, 6], dtype=torch.float16)
        batch_sizes = torch.tensor([3, 2, 1], dtype=torch.int64)
        cpu_out, cpu_lengths = torch._pad_packed_sequence(data, batch_sizes, False, 0, 6)
        npu_out, npu_lengths = torch._pad_packed_sequence(data.npu(), batch_sizes, False, 0, 6)
        self.assertRtolEqual(cpu_out, npu_out.cpu())
        self.assertRtolEqual(cpu_lengths, npu_lengths.cpu())


if __name__ == "__main__":
    run_tests()
