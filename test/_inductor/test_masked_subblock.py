import unittest
import torch
from torch.testing._internal.common_utils import run_tests, instantiate_parametrized_tests
from testutils import TestUtils


class TestMaskedSubblock(TestUtils):
    def eager_function(self, arg0_1):
        repeat = torch.ops.aten.repeat.default(arg0_1, [2, 1, 1])
        slice_3 = torch.ops.aten.slice.Tensor(repeat, 2, 3199, -3199)
        select = torch.ops.aten.select.int(slice_3, 0, 0)
        slice_4 = torch.ops.aten.slice.Tensor(select, 0, 2000, 2001)
        repeat_1 = torch.ops.aten.repeat.default(slice_4, [1199, 1])
        slice_6 = torch.ops.aten.slice.Tensor(select, 0, 2001, 3200)
        copy = torch.ops.aten.copy.default(slice_6, repeat_1)
        slice_10 = torch.ops.aten.slice.Tensor(repeat, 2, 3199, -3199)
        select_1 = torch.ops.aten.select.int(slice_10, 0, 0)
        slice_scatter_1 = torch.ops.aten.slice_scatter.default(select_1, copy, 0, 2001, 3200)
        select_scatter = torch.ops.aten.select_scatter.default(slice_10, slice_scatter_1, 0, 0)
        slice_scatter_2 = torch.ops.aten.slice_scatter.default(repeat, select_scatter, 2, 3199, -3199)
        slice_32 = torch.ops.aten.slice.Tensor(slice_scatter_2, 2, 3199, -3199)
        select_11 = torch.ops.aten.select.int(slice_32, 0, 0)
        slice_35 = torch.ops.aten.slice.Tensor(slice_scatter_2, 2, 3199, -3199)
        select_scatter_1 = torch.ops.aten.select_scatter.default(slice_35, select_11, 0, 0)
        slice_scatter_5 = torch.ops.aten.slice_scatter.default(slice_scatter_2, select_scatter_1, 2, 3199, -3199)
        slice_43 = torch.ops.aten.slice.Tensor(slice_scatter_5, 2, 3199, -3199)
        select_14 = torch.ops.aten.select.int(slice_43, 0, 1)
        slice_44 = torch.ops.aten.slice.Tensor(select_14, 0, 2000, 2001)
        repeat_2 = torch.ops.aten.repeat.default(slice_44, [1199, 1])
        slice_50 = torch.ops.aten.slice.Tensor(slice_scatter_5, 2, 3199, -3199)
        select_15 = torch.ops.aten.select.int(slice_50, 0, 1)
        slice_51 = torch.ops.aten.slice.Tensor(select_15, 0, 2001, 3200)
        copy_2 = torch.ops.aten.copy.default(slice_51, repeat_2)
        slice_55 = torch.ops.aten.slice.Tensor(slice_scatter_5, 2, 3199, -3199)
        select_16 = torch.ops.aten.select.int(slice_55, 0, 1)
        slice_scatter_9 = torch.ops.aten.slice_scatter.default(select_16, copy_2, 0, 2001, 3200)
        select_scatter_2 = torch.ops.aten.select_scatter.default(slice_55, slice_scatter_9, 0, 1)
        slice_scatter_10 = torch.ops.aten.slice_scatter.default(slice_scatter_5, select_scatter_2, 2, 3199, -3199)
        slice_77 = torch.ops.aten.slice.Tensor(slice_scatter_10, 2, 3199, -3199)
        select_26 = torch.ops.aten.select.int(slice_77, 0, 1)
        slice_80 = torch.ops.aten.slice.Tensor(slice_scatter_10, 2, 3199, -3199)
        select_scatter_3 = torch.ops.aten.select_scatter.default(slice_80, select_26, 0, 1)
        slice_scatter_13 = torch.ops.aten.slice_scatter.default(slice_scatter_10, select_scatter_3, 2, 3199, -3199)
        return slice_scatter_13

    @unittest.skip("this test is not supported yet")
    def test_masked_subblock(self):
        arg0_1 = self._generate_tensor((1, 3200, 9598), 'float32')
        std_ret = self.eager_function(arg0_1)
        compiled_op_calc = torch.compile(self.eager_function, backend="inductor", dynamic=False)
        inductor_ret = compiled_op_calc(arg0_1)
        self.assertEqual(std_ret, inductor_ret, atol=1e-1, rtol=1e-1)


instantiate_parametrized_tests(TestMaskedSubblock)

if __name__ == "__main__":
    run_tests()
