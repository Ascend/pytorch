import torch
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestSparseCsr(TestCase):
    def _create_sparse_csr_tensor(self, dtype=torch.float64):
        crow_indices = torch.tensor([0, 2, 4])
        col_indices = torch.tensor([0, 1, 0, 1])
        values = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sparse_csr_cpu = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=dtype)
        sparse_csr_npu = torch.sparse_csr_tensor(crow_indices.npu(), col_indices.npu(), values.npu(),
                                                 dtype=dtype)
        return sparse_csr_npu, sparse_csr_cpu

    def test_sparse_csr_indices_and_values(self):
        sparse_csr_npu, sparse_csr_cpu = self._create_sparse_csr_tensor()
        self.assertRtolEqual(sparse_csr_npu.crow_indices(), sparse_csr_cpu.crow_indices())
        self.assertRtolEqual(sparse_csr_npu.col_indices(), sparse_csr_cpu.col_indices())
        self.assertRtolEqual(sparse_csr_npu.values(), sparse_csr_cpu.values())

    def test_sparse_csr_nnz(self):
        sparse_csr_npu, sparse_csr_cpu = self._create_sparse_csr_tensor()
        self.assertRtolEqual(sparse_csr_npu._nnz(), sparse_csr_cpu._nnz())

    def test_sparse_csr_to_dense(self):
        sparse_csr_npu, sparse_csr_cpu = self._create_sparse_csr_tensor()
        self.assertRtolEqual(sparse_csr_npu.to_dense(), sparse_csr_cpu.to_dense())

    funcs = ["abs", "sgn", "asin", "asinh", "atan", "ceil", "erf", "expm1", "floor", "frac", "log1p", "neg",
             "round", "sin", "sinh", "sqrt", "tan", "tanh", "trunc", "sign", "atanh", "erfinv"]
    float_dtypes = [torch.float, torch.float16]

    @SupportedDevices(['Ascend910B'])
    def test_basic_func(self):
        for dtype in self.float_dtypes:
            for func in self.funcs:
                sparse_csr_npu, sparse_csr_cpu = self._create_sparse_csr_tensor(dtype)
                # e.g. sparse_csr_npu.abs()
                func_str_npu = "sparse_csr_npu." + func + "()"
                func_str_cpu = "sparse_csr_cpu." + func + "()"
                res_cpu = eval(func_str_cpu)
                res_npu = eval(func_str_npu)
                self.assertEqual(res_npu.crow_indices(), res_cpu.crow_indices(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))
                self.assertEqual(res_npu.col_indices(), res_cpu.col_indices(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))
                self.assertEqual(res_npu.values(), res_cpu.values(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))

    @SupportedDevices(['Ascend910B'])
    def test_basic_inplace_func(self):
        for dtype in self.float_dtypes:
            for func in self.funcs:
                sparse_csr_npu, sparse_csr_cpu = self._create_sparse_csr_tensor(dtype)
                # e.g. sparse_csr_npu.abs_()
                func_str_cpu = "sparse_csr_cpu." + func + "_()"
                func_str_npu = "sparse_csr_npu." + func + "_()"
                eval(func_str_cpu)
                eval(func_str_npu)
                self.assertEqual(sparse_csr_npu.crow_indices(), sparse_csr_cpu.crow_indices(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))
                self.assertEqual(sparse_csr_npu.col_indices(), sparse_csr_cpu.col_indices(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))
                self.assertEqual(sparse_csr_npu.values(), sparse_csr_cpu.values(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))

    @SupportedDevices(['Ascend910B'])
    def test_basic_out_func(self):
        for dtype in self.float_dtypes:
            for func in self.funcs:
                sparse_csr_npu, sparse_csr_cpu = self._create_sparse_csr_tensor(dtype)
                res_npu, res_cpu = self._create_sparse_csr_tensor(dtype)
                # e.g. torch.abs(sparse_csr_npu, out = out_npu)
                func_str_cpu = "torch." + func + "(sparse_csr_cpu, out=res_cpu)"
                func_str_npu = "torch." + func + "(sparse_csr_npu, out=res_npu)"
                eval(func_str_cpu)
                eval(func_str_npu)
                self.assertEqual(res_npu.crow_indices(), res_cpu.crow_indices(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))
                self.assertEqual(res_npu.col_indices(), res_cpu.col_indices(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))
                self.assertEqual(res_npu.values(), res_cpu.values(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))

    @SupportedDevices(['Ascend910B'])
    def test_sparse_relu(self):
        for dtype in self.float_dtypes:
            sparse_csr_npu, sparse_csr_cpu = self._create_sparse_csr_tensor(dtype)
            res_cpu = sparse_csr_cpu.relu()
            res_npu = sparse_csr_npu.relu()
            self.assertRtolEqual(res_npu.crow_indices(), res_cpu.crow_indices())
            self.assertRtolEqual(res_npu.col_indices(), res_cpu.col_indices())
            self.assertRtolEqual(res_npu.values(), res_cpu.values())
            sparse_csr_cpu.relu_()
            sparse_csr_npu.relu_()
            self.assertRtolEqual(sparse_csr_npu.crow_indices(), sparse_csr_cpu.crow_indices())
            self.assertRtolEqual(sparse_csr_npu.col_indices(), sparse_csr_cpu.col_indices())
            self.assertRtolEqual(sparse_csr_npu.values(), sparse_csr_cpu.values())
            self.assertRtolEqual(sparse_csr_npu.crow_indices(), res_cpu.crow_indices())
            self.assertRtolEqual(sparse_csr_npu.col_indices(), res_cpu.col_indices())
            self.assertRtolEqual(sparse_csr_npu.values(), res_cpu.values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_angle(self):
        for dtype in self.float_dtypes:
            sparse_csr_npu, sparse_csr_cpu = self._create_sparse_csr_tensor(dtype)
            res_cpu = sparse_csr_cpu.angle()
            res_npu = sparse_csr_npu.angle()
            self.assertRtolEqual(res_npu.crow_indices(), res_cpu.crow_indices())
            self.assertRtolEqual(res_npu.col_indices(), res_cpu.col_indices())
            self.assertRtolEqual(res_npu.values(), res_cpu.values())
            torch.angle(sparse_csr_cpu, out=res_cpu)
            torch.angle(sparse_csr_npu, out=res_npu)
            self.assertRtolEqual(res_npu.crow_indices(), res_cpu.crow_indices())
            self.assertRtolEqual(res_npu.col_indices(), res_cpu.col_indices())
            self.assertRtolEqual(res_npu.values(), res_cpu.values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_signbit(self):
        for dtype in self.float_dtypes:
            sparse_csr_npu, sparse_csr_cpu = self._create_sparse_csr_tensor(dtype)
            res_cpu = sparse_csr_cpu.signbit()
            res_npu = sparse_csr_npu.signbit()
            self.assertRtolEqual(res_npu.crow_indices(), res_cpu.crow_indices())
            self.assertRtolEqual(res_npu.col_indices(), res_cpu.col_indices())
            self.assertRtolEqual(res_npu.values(), res_cpu.values())

if __name__ == "__main__":
    run_tests()
