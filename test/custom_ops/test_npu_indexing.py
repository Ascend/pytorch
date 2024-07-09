import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestNpuIndexing(TestCase):
    def _split_npu_indexing(self,
                            mat,
                            begin,
                            end,
                            strides,
                            begin_mask=0,
                            end_mask=0,
                            ellipsis_mask=0,
                            new_axis_mask=0,
                            shrink_axis_mask=0):
        if begin_mask or end_mask or ellipsis_mask or new_axis_mask or new_axis_mask or shrink_axis_mask:
            raise "Error: Unknown arguments."

        dims = mat.size()
        dim_num = len(dims)
        for i in range(dim_num):
            index_i = [j for j in range(begin[i], end[i], strides[i])]
            mat = torch.index_select(input=mat,
                                     dim=i,
                                     index=torch.tensor(index_i, device="npu"))
        return mat

    def _split_npu_indexing_out(self,
                                mat,
                                begin,
                                end,
                                strides,
                                begin_mask=0,
                                end_mask=0,
                                ellipsis_mask=0,
                                new_axis_mask=0,
                                shrink_axis_mask=0,
                                out=None):
        if begin_mask or end_mask or ellipsis_mask or new_axis_mask or new_axis_mask or shrink_axis_mask:
            raise "Error: Unknown arguments."

        dims = mat.size()
        dim_num = len(dims)
        for i in range(dim_num):
            index_i = [j for j in range(begin[i], end[i], strides[i])]
            mat = torch.index_select(input=mat,
                                     dim=i,
                                     index=torch.tensor(index_i, device="npu"))
        out = mat.clone()
        return out

    def npu_op_exec(self, mat, begin, end, strides):
        output = torch_npu.npu_indexing(mat, begin, end, strides)
        return output

    def split_op_exec(self, mat, begin, end, strides):
        output = self._split_npu_indexing(mat, begin, end, strides)
        return output

    @Dtypes(torch.float, torch.half, torch.int32, torch.uint8, torch.int8, torch.int16, torch.long)
    def test_slice(self, dtype):
        input_data = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).npu().to(dtype)
        split_out = self.split_op_exec(input_data, [0, 0], [2, 2], [1, 1])
        exp_out = self.npu_op_exec(input_data, [0, 0], [2, 2], [1, 1])
        self.assertRtolEqual(split_out, exp_out)

        input_data = torch.tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                                   [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                                   [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]]).npu().to(dtype)
        split_out = self.split_op_exec(input_data, [0, 0, 0], [2, 2, 2], [1, 1, 1])
        exp_out = self.npu_op_exec(input_data, [0, 0, 0], [2, 2, 2], [1, 1, 1])
        self.assertRtolEqual(split_out, exp_out)


if __name__ == "__main__":
    run_tests()
