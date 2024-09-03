import unittest
from numbers import Number
import torch
from torch.testing import make_tensor
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestSparseCoo(TestCase):
    def _create_sparse_coo_tensor(self):
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
        values = torch.tensor([1, 2, 4])
        sparse_coo_cpu = torch.sparse_coo_tensor(indices=indices, values=values, size=[4, 4])
        sparse_coo_npu = torch.sparse_coo_tensor(indices=indices.npu(), values=values.npu(), size=[4, 4])
        return sparse_coo_npu, sparse_coo_cpu

    def test_sparse_coo_nnz(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        self.assertRtolEqual(sparse_coo_npu._nnz(), sparse_coo_cpu._nnz())

    def test_sparse_coo_indices_and_values(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        self.assertRtolEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())
        self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())

    def test_sparse_coo_dim(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        self.assertRtolEqual(sparse_coo_npu.sparse_dim(), sparse_coo_cpu.sparse_dim())
        self.assertRtolEqual(sparse_coo_npu.dense_dim(), sparse_coo_cpu.dense_dim())

    def test_sparse_coo_is_coalesced(self):
        sparse_coo_npu, _ = self._create_sparse_coo_tensor()
        self.assertEqual(sparse_coo_npu.is_coalesced(), False)
        sparse_coo_npu._coalesced_(True)
        self.assertEqual(sparse_coo_npu.is_coalesced(), True)

    def test_sparse_coo_neg(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        sparse_coo_cpu_neg = sparse_coo_cpu.neg()
        sparse_coo_npu_neg = sparse_coo_npu.neg()
        self.assertRtolEqual(sparse_coo_cpu_neg._values(), sparse_coo_npu_neg._values())

    def test_sparse_coo_copy_(self):
        sparse_coo_npu, _ = self._create_sparse_coo_tensor()
        copy_sparse_tensor = torch.rand(sparse_coo_npu.shape).to(sparse_coo_npu.dtype).to_sparse().npu()
        copy_sparse_tensor.copy_(sparse_coo_npu)
        self.assertRtolEqual(copy_sparse_tensor._values(), sparse_coo_npu._values())
        self.assertRtolEqual(copy_sparse_tensor._indices(), sparse_coo_npu._indices())

    def test_sparse_coo_dense(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        self.assertRtolEqual(sparse_coo_npu.to_dense(), sparse_coo_cpu.to_dense())

    dtypes = [torch.float, torch.float16, torch.int32]

    @SupportedDevices(['Ascend910B'])
    def test_sparse_coalesce(self):
        for dtype in self.dtypes:
            i = torch.Tensor([[1, 2, 1, 2], [2, 2, 2, 3]])
            v = torch.Tensor([1, 2, 3, 4]).to(dtype)
            sparse_coo_cpu = torch.sparse_coo_tensor(indices=i, values=v, size=[4, 4]).coalesce()
            sparse_coo_npu = torch.sparse_coo_tensor(indices=i.npu(), values=v.npu(), size=[4, 4]).coalesce()
            self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())
            self.assertRtolEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())

            v2 = torch.Tensor([[1, 2], [2, 3], [3, 3], [4, 9]]).to(dtype)
            sparse_coo_cpu2 = torch.sparse_coo_tensor(indices=i, values=v2, size=[4, 4, 2]).coalesce()
            sparse_coo_npu2 = torch.sparse_coo_tensor(indices=i.npu(), values=v2.npu(), size=[4, 4, 2]).coalesce()
            self.assertRtolEqual(sparse_coo_npu2._values(), sparse_coo_cpu2._values())
            self.assertRtolEqual(sparse_coo_npu2._indices(), sparse_coo_cpu2._indices())

            i2 = torch.Tensor([[1, 2, 1, 2], [2, 2, 2, 3], [2, 2, 2, 3]])
            sparse_coo_cpu3 = torch.sparse_coo_tensor(indices=i2, values=v, size=[4, 4, 4]).coalesce()
            sparse_coo_npu3 = torch.sparse_coo_tensor(indices=i2.npu(), values=v.npu(), size=[4, 4, 4]).coalesce()
            self.assertRtolEqual(sparse_coo_npu3._values(), sparse_coo_cpu3._values())
            self.assertRtolEqual(sparse_coo_npu3._indices(), sparse_coo_cpu3._indices())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_coalesce_large_shape_case1(self):
        for dtype in self.dtypes:
            i = torch.randint(0, 400, (2, 458409)).to(torch.int32)
            v = torch.randn(458409,).to(dtype)
            if dtype == torch.float16:
                sparse_coo_cpu = torch.sparse_coo_tensor(indices=i, values=v.to(torch.float)).coalesce()
                sparse_coo_npu = torch.sparse_coo_tensor(indices=i.npu(), values=v.npu()).coalesce()
                self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values().to(torch.float16))
            else:
                sparse_coo_cpu = torch.sparse_coo_tensor(indices=i, values=v).coalesce()
                sparse_coo_npu = torch.sparse_coo_tensor(indices=i.npu(), values=v.npu()).coalesce()
                self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())
            self.assertRtolEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_coalesce_large_shape_case2(self):
        for dtype in self.dtypes:
            i = torch.randint(0, 400, (2, 5275)).to(torch.int32)
            v = torch.randn(5275,).to(dtype)
            if dtype == torch.float16:
                sparse_coo_cpu = torch.sparse_coo_tensor(indices=i, values=v.to(torch.float)).coalesce()
                sparse_coo_npu = torch.sparse_coo_tensor(indices=i.npu(), values=v.npu()).coalesce()
                self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values().to(torch.float16))
            else:
                sparse_coo_cpu = torch.sparse_coo_tensor(indices=i, values=v).coalesce()
                sparse_coo_npu = torch.sparse_coo_tensor(indices=i.npu(), values=v.npu()).coalesce()
                self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())
            self.assertRtolEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_add(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        input1 = torch.rand(4, 4)

        with self.assertRaises(RuntimeError):
            sparse_coo_npu + input1.npu()

        output_cpu1 = input1 + sparse_coo_cpu
        output_npu1 = input1.npu() + sparse_coo_npu
        self.assertRtolEqual(output_cpu1, output_npu1)

        output_cpu2 = input1.to_sparse() + sparse_coo_cpu
        output_npu2 = input1.to_sparse().npu() + sparse_coo_npu
        self.assertRtolEqual(output_npu2._values(), output_cpu2._values())
        self.assertRtolEqual(output_npu2._indices(), output_cpu2._indices())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_sum(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        self.assertRtolEqual(sparse_coo_npu.sum(), sparse_coo_cpu.sum())

    def test_sparse_max(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        self.assertRtolEqual(sparse_coo_npu.max(), sparse_coo_cpu.to_dense().max())

    def setUp(self):
        TestCase.setUp(self)
        self.index_tensor = lambda *args, **kwargs: torch.tensor(*args, **kwargs, dtype=torch.int64)

        def sparse_tensor_factory(*args, **kwargs):
            return torch.sparse_coo_tensor(*args, **kwargs)

        self.sparse_tensor = sparse_tensor_factory

    def genSparseTensor(self, size, sparse_dim, nnz, is_uncoalesced, device, dtype):
        # Assert not given impossible combination, where the sparse dims have
        # empty numel, but nnz > 0 makes the indices containing values.
        assert all(size[d] > 0 for d in range(sparse_dim)) or nnz == 0, 'invalid arguments'

        v_size = [nnz] + list(size[sparse_dim:])
        v = make_tensor(v_size, device=device, dtype=dtype, low=-1, high=1)
        i = torch.rand(sparse_dim, nnz, device=device)
        i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
        i = i.to(torch.long)
        if is_uncoalesced:
            i1 = i[:, :(nnz // 2), ...]
            i2 = i[:, :((nnz + 1) // 2), ...]
            i = torch.cat([i1, i2], 1)
        x = torch.sparse_coo_tensor(i, v, torch.Size(size), dtype=dtype, device=device)

        if not is_uncoalesced:
            x = x.coalesce()
        else:
            # FIXME: `x` is a sparse view of `v`. Currently rebase_history for
            #        sparse views is not implemented, so this workaround is
            #        needed for inplace operations done on `x`, e.g., copy_().
            #        Remove after implementing something equivalent to CopySlice
            #        for sparse views.
            # NOTE: We do clone() after detach() here because we need to be able to change size/storage of x afterwards
            x = x.detach().clone()._coalesced_(False)
        return x, x._indices().clone(), x._values().clone()

    def _gen_sparse(self, sparse_dim, nnz, with_size, dtype, device):
        if isinstance(with_size, Number):
            with_size = [with_size] * sparse_dim
        x, i, v = self.genSparseTensor(with_size, sparse_dim, nnz, False, dtype=dtype, device=device)
        return x, i, v

    @SupportedDevices(['Ascend910B'])
    def test_basic(self):
        device = "npu"

        def test_shape(sparse_dims, nnz, with_size, dtype):
            if isinstance(with_size, Number):
                with_size = [with_size] * sparse_dims
            x, i, v = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device)
            self.assertEqual(i, x._indices())
            self.assertEqual(v, x._values())
            self.assertEqual(x.ndimension(), len(with_size))
            self.assertEqual(x.coalesce()._nnz(), nnz if x.is_coalesced() else nnz // 2)
            self.assertEqual(list(x.size()), with_size)

            self.assertEqual(x.indices(), x._indices())
            self.assertEqual(x.values(), x._values())

        for dtype in self.dtypes:
            test_shape(3, 10, 100, dtype)
            test_shape(3, 10, [100, 100, 100], dtype)
            test_shape(3, 10, [100, 100, 100, 5, 5, 5, 0], dtype)
            test_shape(3, 0, [0, 0, 100, 5, 5, 5, 0], dtype)

            # Make sure that coalesce handles duplicate indices correctly
            i = self.index_tensor([[9, 0, 0, 0, 8, 1, 1, 1, 2, 7, 2, 2, 3, 4, 6, 9]], device=device)
            v = torch.tensor([[idx ** 2, idx] for idx in range(i.size(1))], dtype=dtype, device=device)
            x = self.sparse_tensor(i, v, torch.Size([10, 2]), dtype=dtype, device=device)
            self.assertEqual(x.coalesce()._nnz(), 9)

    def _create_sparse_coo_coalesced_tensor(self, dtype):
        i = torch.Tensor([[1, 2, 3, 2], [2, 2, 2, 3]])
        v = torch.Tensor([2, -2, 3, -4]).to(dtype)
        sparse_coo_cpu = torch.sparse_coo_tensor(indices=i, values=v, size=[4, 4]).coalesce()
        sparse_coo_npu = torch.sparse_coo_tensor(indices=i.npu(), values=v.npu(), size=[4, 4]).coalesce()
        return sparse_coo_npu, sparse_coo_cpu

    funcs = ["abs", "asinh", "asin", "atan", "atanh", "ceil", "deg2rad", "erfinv", "erf", "expm1", "floor", "frac", "log1p",
             "round", "rad2deg", "sign", "sgn", "sin", "sinh", "sqrt", "tan", "tanh", "trunc", "nan_to_num", "neg"]
    float_dtypes = [torch.float, torch.float16]

    @SupportedDevices(['Ascend910B'])
    def test_basic_func(self):
        for dtype in self.float_dtypes:
            for func in self.funcs:
                sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(dtype)
                # e.g. sparse_coo_npu.abs()
                func_str_npu = "sparse_coo_npu." + func + "()"
                func_str_cpu = "sparse_coo_cpu." + func + "()"
                res_cpu = eval(func_str_cpu)
                res_npu = eval(func_str_npu)
                self.assertEqual(res_npu._indices(), res_cpu._indices(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))
                self.assertEqual(res_npu._values(), res_cpu._values(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))

    @SupportedDevices(['Ascend910B'])
    def test_basic_inplace_func(self):
        for dtype in self.float_dtypes:
            for func in self.funcs:
                sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(dtype)
                # e.g. sparse_coo_npu.abs_()
                func_str_cpu = "sparse_coo_cpu." + func + "_()"
                func_str_npu = "sparse_coo_npu." + func + "_()"
                eval(func_str_cpu)
                eval(func_str_npu)
                self.assertEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))
                self.assertEqual(sparse_coo_npu._values(), sparse_coo_cpu._values(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))

    @SupportedDevices(['Ascend910B'])
    def test_basic_out_func(self):
        for dtype in self.float_dtypes:
            for func in self.funcs:
                sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(dtype)
                res_npu, res_cpu = self._create_sparse_coo_coalesced_tensor(dtype)
                # torch.abs(sparse_coo_npu, out = out_npu)
                func_str_cpu = "torch." + func + "(sparse_coo_cpu, out=res_cpu)"
                func_str_npu = "torch." + func + "(sparse_coo_npu, out=res_npu)"
                eval(func_str_cpu)
                eval(func_str_npu)
                self.assertEqual(res_npu._indices(), res_cpu._indices(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))
                self.assertEqual(res_npu._values(), res_cpu._values(),
                                 message="now in " + func_str_npu + " with dtype " + str(dtype))

    @SupportedDevices(['Ascend910B'])
    def test_sparse_relu(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = sparse_coo_cpu.relu()
        res_npu = sparse_coo_npu.relu()
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())
        sparse_coo_cpu.relu_()
        sparse_coo_npu.relu_()
        self.assertRtolEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())
        self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())
        self.assertRtolEqual(sparse_coo_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(sparse_coo_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_signbit(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_npu, res_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = torch.signbit(sparse_coo_cpu)
        res_npu = torch.signbit(sparse_coo_npu)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())
        torch.signbit(sparse_coo_cpu, out=res_cpu)
        torch.signbit(sparse_coo_npu, out=res_npu)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_isposinf(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = torch.isposinf(sparse_coo_cpu)
        res_npu = torch.isposinf(sparse_coo_npu)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())
        torch.isposinf(sparse_coo_cpu, out=res_cpu)
        torch.isposinf(sparse_coo_npu, out=res_npu)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_isneginf(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = torch.isneginf(sparse_coo_cpu)
        res_npu = torch.isneginf(sparse_coo_npu)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())
        torch.isneginf(sparse_coo_cpu, out=res_cpu)
        torch.isneginf(sparse_coo_npu, out=res_npu)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_isnan(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = torch.isnan(sparse_coo_cpu)
        res_npu = torch.isnan(sparse_coo_npu)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_isinf(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = torch.isinf(sparse_coo_cpu)
        res_npu = torch.isinf(sparse_coo_npu)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_threshold_backward(self):
        # threshold_backward
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        sparse_coo_cpu_1 = sparse_coo_cpu
        sparse_coo_npu_1 = sparse_coo_npu
        res_cpu = torch.ops.aten.threshold_backward(sparse_coo_cpu, sparse_coo_cpu_1, 2)
        res_npu = torch.ops.aten.threshold_backward(sparse_coo_npu, sparse_coo_npu_1, 2)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())
        # threshold_backward.grad_input
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        sparse_coo_cpu_1 = sparse_coo_cpu
        sparse_coo_npu_1 = sparse_coo_npu
        torch.ops.aten.threshold_backward(sparse_coo_cpu, sparse_coo_cpu_1, 2, grad_input=res_cpu)
        torch.ops.aten.threshold_backward(sparse_coo_npu, sparse_coo_npu_1, 2, grad_input=res_npu)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_conj_physical(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_npu, res_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        torch.conj_physical(sparse_coo_cpu, out=res_cpu)
        torch.conj_physical(sparse_coo_npu, out=res_npu)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse__sparse_broadcast_to(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = torch._sparse_broadcast_to(sparse_coo_cpu, [4, 4])
        res_npu = torch._sparse_broadcast_to(sparse_coo_npu, [4, 4])
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_cat(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        sparse_coo_cpu_1 = sparse_coo_cpu
        sparse_coo_npu_1 = sparse_coo_npu
        res_cpu = torch.cat((sparse_coo_cpu_1, sparse_coo_cpu), 0)
        res_npu = torch.cat((sparse_coo_npu_1, sparse_coo_npu), 0)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_clone(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = torch.clone(sparse_coo_cpu)
        res_npu = torch.clone(sparse_coo_npu)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_copy_sparse_to_sparse_(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        sparse_coo_cpu_1 = sparse_coo_cpu
        sparse_coo_npu_1 = sparse_coo_npu
        torch.ops.aten.copy_sparse_to_sparse_(sparse_coo_cpu_1, sparse_coo_cpu)
        torch.ops.aten.copy_sparse_to_sparse_(sparse_coo_npu_1, sparse_coo_npu)
        self.assertRtolEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())
        self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())
        self.assertRtolEqual(sparse_coo_npu_1._indices(), sparse_coo_cpu_1._indices())
        self.assertRtolEqual(sparse_coo_npu_1._values(), sparse_coo_cpu_1._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_div(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        # div.Tensor
        res_cpu = torch.div(sparse_coo_cpu, 2)
        res_npu = torch.div(sparse_coo_npu, 2)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())
        # div_
        sparse_coo_cpu.div_(2)
        sparse_coo_npu.div_(2)
        self.assertRtolEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())
        self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())
        self.assertRtolEqual(sparse_coo_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(sparse_coo_npu._values(), res_cpu._values())
        # div.out
        torch.div(sparse_coo_cpu, 2, out=res_cpu)
        torch.div(sparse_coo_npu, 2, out=res_npu)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        # div.Tensor_mode
        res_cpu = torch.div(sparse_coo_cpu, 2, rounding_mode=None)
        res_npu = torch.div(sparse_coo_npu, 2, rounding_mode=None)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())
        # div_.Tensor_mode
        sparse_coo_cpu.div_(2, rounding_mode=None)
        sparse_coo_npu.div_(2, rounding_mode=None)
        self.assertRtolEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())
        self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())
        self.assertRtolEqual(sparse_coo_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(sparse_coo_npu._values(), res_cpu._values())
        # div.out_mode
        torch.div(sparse_coo_cpu, 2, out=res_cpu, rounding_mode=None)
        torch.div(sparse_coo_npu, 2, out=res_npu, rounding_mode=None)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse__sparse_coo_tensor_with_dims(self):
        res_cpu = torch.ops.aten._sparse_coo_tensor_with_dims(2, 0, [4, 4], layout=1)
        res_npu = torch.ops.aten._sparse_coo_tensor_with_dims(2, 0, [4, 4], device="npu", layout=1)  # layout=1:sparse
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_empty(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = torch.empty([4, 4], layout=torch.sparse_coo, memory_format=torch.contiguous_format)
        res_npu = torch.empty([4, 4], layout=torch.sparse_coo, device="npu", memory_format=torch.contiguous_format)
        self.assertEqual(res_npu._indices(), res_cpu._indices())
        self.assertEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_sparse_resize_and_clear_(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = sparse_coo_cpu.sparse_resize_and_clear_([4, 4], 2, 0)
        res_npu = sparse_coo_npu.sparse_resize_and_clear_([4, 4], 2, 0)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_empty_like(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = torch.empty_like(sparse_coo_cpu)
        res_npu = torch.empty_like(sparse_coo_npu)
        self.assertEqual(res_npu._indices(), res_cpu._indices())
        self.assertEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_floor_divide(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        # floor_divide
        res_cpu = torch.floor_divide(sparse_coo_cpu, 2)
        res_npu = torch.floor_divide(sparse_coo_npu, 2)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())
        # floor_divide_.Tensor
        sparse_coo_cpu.floor_divide_(2)
        sparse_coo_npu.floor_divide_(2)
        self.assertRtolEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())
        self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())
        self.assertRtolEqual(sparse_coo_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(sparse_coo_npu._values(), res_cpu._values())
        # floor_divide.out
        torch.floor_divide(sparse_coo_cpu, 2, out=res_cpu)
        torch.floor_divide(sparse_coo_npu, 2, out=res_npu)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_narrow_copy(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = sparse_coo_cpu.narrow_copy(0, 0, 1)
        res_npu = sparse_coo_npu.narrow_copy(0, 0, 1)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_permute(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = sparse_coo_cpu.permute([0, 1])
        res_npu = sparse_coo_npu.permute([0, 1])
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_sum(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = sparse_coo_cpu.sum([0])
        res_npu = sparse_coo_npu.sum([0])
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_unsqueeze(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = sparse_coo_cpu.unsqueeze(0)
        res_npu = sparse_coo_npu.unsqueeze(0)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_zeros(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        torch.zeros([4, 4], out=sparse_coo_cpu)
        torch.zeros([4, 4], out=sparse_coo_npu)
        self.assertEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())
        self.assertEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_native_norm(self):
        # native_norm
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = torch.native_norm(sparse_coo_cpu)
        res_npu = torch.native_norm(sparse_coo_npu)
        self.assertRtolEqual(res_npu, res_cpu)
        # native_norm.ScalarOpt_dim_dtype
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = torch.native_norm(sparse_coo_cpu, 1, [0, 1], False, None)
        res_npu = torch.native_norm(sparse_coo_npu, 1, [0, 1], False, None)
        self.assertRtolEqual(res_npu, res_cpu)

    @SupportedDevices(['Ascend910B'])
    def test_sparse_norm(self):
        # norm.ScalarOpt_dim_dtype
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = torch.norm(sparse_coo_cpu, None, [0, 1], False, dtype=None)
        res_npu = torch.norm(sparse_coo_npu, None, [0, 1], False, dtype=None)
        self.assertRtolEqual(res_npu, res_cpu)
        # norm.ScalarOpt_dim
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = torch.norm(sparse_coo_cpu, None, [0, 1], False)
        res_npu = torch.norm(sparse_coo_npu, None, [0, 1], False)
        self.assertRtolEqual(res_npu, res_cpu)

    @SupportedDevices(['Ascend910B'])
    def test_sparse_resize_as_sparse_(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        i = torch.Tensor([[1, 2, 3, 2], [2, 2, 2, 3]])
        v = torch.Tensor([2, -2, 3, -4]).to(torch.float)
        sparse_coo_cpu_1 = torch.sparse_coo_tensor(indices=i, values=v, size=[8, 4]).coalesce()
        sparse_coo_npu_1 = torch.sparse_coo_tensor(indices=i.npu(), values=v.npu(), size=[8, 4]).coalesce()
        sparse_coo_cpu.resize_as_sparse_(sparse_coo_cpu_1)
        sparse_coo_npu.resize_as_sparse_(sparse_coo_npu_1)
        self.assertRtolEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())
        self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_zero_(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        sparse_coo_cpu.zero_()
        sparse_coo_npu.zero_()
        self.assertRtolEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())
        self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_sparse_resize_(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        sparse_coo_cpu.sparse_resize_([8, 4], 2, 0)
        sparse_coo_npu.sparse_resize_([8, 4], 2, 0)
        self.assertRtolEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())
        self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse__dimI(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        sparse_coo_cpu._dimI()
        sparse_coo_npu._dimI()
        self.assertEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())
        self.assertEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse__dimV(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        sparse_coo_cpu._dimV()
        sparse_coo_npu._dimV()
        self.assertEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())
        self.assertEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_indices_values(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        self.assertEqual(sparse_coo_npu.indices(), sparse_coo_cpu.indices())
        self.assertEqual(sparse_coo_npu.values(), sparse_coo_cpu.values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse__to_sparse(self):
        tensor_npu = torch.ones([4]).npu()
        tensor_cpu = torch.ones([4])
        res_cpu = tensor_cpu._to_sparse()
        res_npu = tensor_npu._to_sparse()
        self.assertEqual(res_npu.indices(), res_cpu.indices())
        self.assertEqual(res_npu.values(), res_cpu.values())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_any(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = sparse_coo_cpu.any()
        res_npu = sparse_coo_npu.any()
        self.assertRtolEqual(res_npu, res_cpu)

    @SupportedDevices(['Ascend910B'])
    def test_sparse_pow(self):
        # pow.Tensor_Scalar
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = sparse_coo_cpu.pow(2)
        res_npu = sparse_coo_npu.pow(2)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())
        # pow.Tensor_Scalar_out
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_coalesced_tensor(torch.float)
        res_cpu = sparse_coo_cpu.pow(2)
        res_npu = sparse_coo_npu.pow(2)
        self.assertRtolEqual(res_npu._indices(), res_cpu._indices())
        self.assertRtolEqual(res_npu._values(), res_cpu._values())

if __name__ == "__main__":
    run_tests()
