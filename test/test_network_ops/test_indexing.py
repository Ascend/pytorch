import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestIndexing(TestCase):
    @Dtypes(torch.float, torch.half, torch.int32, torch.uint8, torch.int8, torch.int16, torch.long)
    def test_indexing(self, device, dtype):
        input1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype).to("npu")
        expect_output = torch.tensor([[1, 2], [5, 6]], dtype=dtype)
        output = torch_npu.npu_indexing(input1, [0, 0], [2, 2], [1, 1])
        self.assertRtolEqual(expect_output, output.cpu())


if __name__ == "__main__":
    run_tests()
