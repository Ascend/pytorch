import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestNpuSlice(TestCase):
    def split_npu_slice(self, input1, offset, sizes):
        input_dim = input1.size()
        num_dim = len(input_dim)
        for i in range(num_dim):
            input_index = [j for j in range(offset[i], offset[i] + sizes[i])]
            input1 = torch.index_select(input=input1,
                                        dim=i,
                                        index=torch.tensor(input_index, device="npu"))
        return input1

    def split_npu_slice_out(self, input1, offset, sizes, out=None):
        input_dim = input1.size()
        num_dim = len(input_dim)
        for i in range(num_dim):
            input_index = [j for j in range(offset[i], offset[i] + sizes[i])]
            input1 = torch.index_select(input=input1,
                                        dim=i,
                                        index=torch.tensor(input_index, device="npu"))
        out = input1.clone()
        return out

    def npu_op_exec(self, input1, offset, sizes):
        output = torch_npu.npu_slice(input1, offset, sizes)
        return output

    def split_op_exec(self, input1, offset, sizes):
        output = self.split_npu_slice(input1, offset, sizes)
        return output

    @Dtypes(torch.float, torch.half, torch.int32, torch.uint8, torch.int8, torch.int16, torch.long)
    def test_slice(self, dtype):
        input_data = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).npu().to(dtype)
        split_out = self.split_op_exec(input_data, [0, 0], [2, 2])
        exp_out = self.npu_op_exec(input_data, [0, 0], [2, 2])
        self.assertRtolEqual(split_out, exp_out)

        input_data = torch.tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                                   [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                                   [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]]).npu().to(dtype)
        split_out = self.split_op_exec(input_data, [0, 0, 0], [2, 2, 2])
        exp_out = self.npu_op_exec(input_data, [0, 0, 0], [2, 2, 2])
        self.assertRtolEqual(split_out, exp_out)


if __name__ == '__main__':
    run_tests()
