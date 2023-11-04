import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

device = 'npu:0'
torch.npu.set_device(device)


class TestParallelism(TestCase):
    def test_set_num_threads(self):
        torch.set_num_threads(2)

    def test_get_num_threads(self):
        output = torch.get_num_threads()
        print(output)

    def test_set_num_interop_threads(self):
        torch.set_num_interop_threads(2)

    def test_get_num_interop_threads(self):
        output = torch.get_num_interop_threads()
        print(output)


if __name__ == "__main__":
    run_tests()
