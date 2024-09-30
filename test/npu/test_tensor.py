import itertools
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import DeterministicGuard

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestTensor(TestCase):

    def test_narrow_empty(self, device="npu"):
        x = torch.randn(2, 3, 4).to(device=device)
        for d in range(x.dim()):
            y = x.narrow(d, x.size(d), 0)
            sz = list(x.size())
            sz[d] = 0
            self.assertEqual(sz, y.size())

    def test_tensor_set(self):
        t1 = torch.Tensor().npu()
        t2 = torch.Tensor(3, 4, 9, 10).uniform_().npu()
        t1.set_(t2)
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        size = torch.Size([9, 3, 4, 10])
        t1.set_(t2.storage(), 0, size)
        self.assertEqual(t1.size(), size)
        t1.set_(t2.storage(), 0, tuple(size))
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), (120, 40, 10, 1))
        stride = (10, 360, 90, 1)
        t1.set_(t2.storage(), 0, size, stride)
        self.assertEqual(t1.stride(), stride)
        t1.set_(t2.storage(), 0, size=size, stride=stride)
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), stride)

        # test argument names
        t1 = torch.Tensor().npu()
        # 1. case when source is tensor
        t1.set_(source=t2)
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        # 2. case when source is storage
        t1.set_(source=t2.storage())
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        # 3. case when source is storage, and other args also specified
        t1.set_(source=t2.storage(), storage_offset=0, size=size, stride=stride)
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), stride)

        t1 = torch.tensor([True, True], dtype=torch.bool).npu()
        t2 = torch.tensor([False, False], dtype=torch.bool).npu()
        t1.set_(t2)
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)

    def test_tensor_set_with_new_storage(self):
        cpu_t_1 = torch.range(0, 15, dtype=torch.float)
        cpu_t_2 = cpu_t_1.reshape((4, 4))
        cpu_s = cpu_t_2.untyped_storage()
        npu_tensor_1 = cpu_t_1.npu()
        npu_tensor_2 = cpu_t_2.npu()

        npu_s_1 = torch.UntypedStorage(cpu_s.size(), device=torch.device('npu'))
        npu_s_1.copy_(cpu_s, non_blocking=True)
        npu_t_1 = torch.empty([], dtype=torch.float, device=torch.device('npu'))
        npu_t_1.set_(npu_s_1)
        self.assertEqual(npu_tensor_1.cpu(), npu_t_1.cpu())
        self.assertEqual(torch_npu.get_npu_format(torch_npu.npu_format_cast(npu_t_1, 29)), 2)

        npu_s_2 = torch.UntypedStorage(cpu_s.size(), device=torch.device('npu'))
        npu_s_2.copy_(cpu_s, non_blocking=True)
        npu_t_2 = torch.empty([], dtype=torch.float, device=torch.device('npu'))
        npu_t_2.set_(npu_s_2, 0, [4, 4], [4, 1])
        self.assertEqual(npu_tensor_2.cpu(), npu_t_2.cpu())
        self.assertEqual(torch_npu.get_npu_format(torch_npu.npu_format_cast(npu_t_2, 29)), 2)

        npu_s_3 = torch.UntypedStorage(cpu_s.size(), device=torch.device('npu'))
        npu_s_3.copy_(cpu_s, non_blocking=True)
        npu_t_3 = torch.empty([], dtype=torch.float, device=torch.device('npu'))
        npu_t_3.set_(npu_s_3, 0, [4, 2], [4, 1])
        self.assertEqual(npu_tensor_2[:, :2].cpu(), npu_t_3.cpu())

    @Dtypes(torch.half, torch.float)
    def test_set_storage(self, dtype):
        device = torch.device("npu")
        a = make_tensor((4, 5, 3), dtype=dtype, device=device, low=-9, high=9)
        a_s = a.storage()
        b = torch.tensor([], device=device, dtype=dtype).set_(a_s).reshape(a.size())
        self.assertEqual(a.cpu(), b.cpu())
        c = torch.tensor([], device=device, dtype=dtype).set_(a_s.untyped()).reshape(a.size())
        self.assertEqual(a.cpu(), c.cpu())

        x = torch.randn((1, 2, 3), dtype=dtype, device=device)

        def inplace():
            y = torch.randn((1, 2, 3), dtype=dtype, device=device)
            return y

        inplace().set_()
        inplace().set_(x)
        inplace().set_(x.storage())
        inplace().set_(x.storage(), x.storage_offset(), x.size(), x.stride())

    @Dtypes(torch.half, torch.float)
    def test_cat_all_dtypes_and_devices(self, device, dtype):
        x = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)

        expected1 = torch.tensor([[1, 2], [3, 4], [1, 2], [3, 4]], dtype=dtype, device=device)
        self.assertEqual(torch.cat((x, x), 0).to('cpu'), expected1.to('cpu'))

        expected2 = torch.tensor([[1, 2, 1, 2], [3, 4, 3, 4]], dtype=dtype, device=device)
        self.assertEqual(torch.cat((x, x), 1).to('cpu'), expected2.to('cpu'))

    def test_cat_mem_overlap(self, device="npu"):
        x = torch.rand((1, 3)).to(device).expand((6, 3))
        y = torch.rand((3, 3)).to(device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            torch.cat([y, y], out=x)

    def test_cat(self, device="npu"):
        SIZE = 10
        for dim in range(-3, 3):
            pos_dim = dim if dim >= 0 else 3 + dim
            x = torch.rand(13, SIZE, SIZE).to(device).transpose(0, pos_dim)
            y = torch.rand(17, SIZE, SIZE).to(device).transpose(0, pos_dim)
            z = torch.rand(19, SIZE, SIZE).to(device).transpose(0, pos_dim)

            res1 = torch.cat((x, y, z), dim)
            self.assertEqual(res1.narrow(pos_dim, 0, 13).to('cpu'), x.to('cpu'))
            self.assertEqual(res1.narrow(pos_dim, 13, 17).to('cpu'), y.to('cpu'))
            self.assertEqual(res1.narrow(pos_dim, 30, 19).to('cpu'), z.to('cpu'))

        x = torch.randn(20, SIZE, SIZE).to(device)
        self.assertEqual(torch.cat(torch.split(x, 7)).to('cpu'), x.to('cpu'))
        self.assertEqual(torch.cat(torch.chunk(x, 7)).to('cpu'), x.to('cpu'))

        y = torch.randn(1, SIZE, SIZE).to(device)
        z = torch.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))

    def test_zeros(self, device="npu"):
        res1 = torch.zeros(100, 100, device=device)
        res2 = torch.tensor((), device=device)
        torch.zeros(100, 100, device=device, out=res2)

        self.assertEqual(res1.to('cpu'), res2.to('cpu'))

        boolTensor = torch.zeros(2, 2, device=device, dtype=torch.bool)
        expected = torch.tensor([[False, False], [False, False]],
                                device=device, dtype=torch.bool)
        self.assertEqual(boolTensor.to('cpu'), expected.to('cpu'))

        halfTensor = torch.zeros(1, 1, device=device, dtype=torch.half)
        expected = torch.tensor([[0.]], device=device, dtype=torch.float16)
        self.assertEqual(halfTensor.to('cpu'), expected.to('cpu'))

        bfloat16Tensor = torch.zeros(1, 1, device=device, dtype=torch.half)
        expected = torch.tensor([[0.]], device=device, dtype=torch.half)
        self.assertEqual(bfloat16Tensor.to('cpu'), expected.to('cpu'))

    def test_zeros_out(self, device="npu"):
        shape = (3, 4)
        out = torch.zeros(shape, device=device)
        torch.zeros(shape, device=device, out=out)

        # change the dtype, layout, device
        with self.assertRaises(RuntimeError):
            torch.zeros(shape, device=device, dtype=torch.int64, out=out)
        with self.assertRaises(RuntimeError):
            torch.zeros(shape, device=device, layout=torch.sparse_coo, out=out)

        # leave them the same
        self.assertEqual(torch.zeros(shape, device=device).to('cpu'),
                         torch.zeros(shape, device=device, dtype=out.dtype, out=out).to('cpu'))
        self.assertEqual(torch.zeros(shape, device=device).to('cpu'),
                         torch.zeros(shape, device=device, layout=torch.strided, out=out).to('cpu'))
        self.assertEqual(torch.zeros(shape, device=device).to('cpu'),
                         torch.zeros(shape, device=device, out=out).to('cpu'))

    def test_ones(self, device="npu"):
        res1 = torch.ones(100, 100, device=device)
        res2 = torch.tensor((), device=device)
        torch.ones(100, 100, device=device, out=res2)
        self.assertEqual(res1.to('cpu'), res2.to('cpu'))

        # test boolean tensor
        res1 = torch.ones(1, 2, device=device, dtype=torch.bool)
        expected = torch.tensor([[True, True]], device=device, dtype=torch.bool)
        self.assertEqual(res1.to('cpu'), expected.to('cpu'))

    def test_empty_strided(self, device="npu"):
        for shape in [(2, 3, 4), (0, 2, 0)]:
            # some of these cases are pretty strange, just verifying that if as_strided
            # allows them then empty_strided can as well.
            for strides in [(12, 4, 1), (0, 0, 0)]:
                empty_strided = torch.empty_strided(shape, strides, device=device)
                # as_strided checks the storage size is big enough to support such a strided tensor;
                # instead of repeating this calculation, we just use empty_strided which does the same
                # calculation when setting the storage size.
                as_strided = torch.empty(empty_strided.storage().size(),
                                         device=device).as_strided(shape, strides)
                self.assertEqual(empty_strided.shape, as_strided.shape)
                self.assertEqual(empty_strided.stride(), as_strided.stride())

    def test_empty_tensor_props(self, device="npu"):
        sizes = [(0,), (0, 3), (5, 0), (5, 0, 3, 0, 2), (0, 3, 0, 2), (0, 5, 0, 2, 0)]
        for size in sizes:
            x = torch.empty(tuple(size), device=device)
            self.assertEqual(size, x.shape)
            self.assertTrue(x.is_contiguous())
            size_ones_instead_of_zeros = (x if x != 0 else 1 for x in size)
            y = torch.empty(tuple(size_ones_instead_of_zeros), device=device)
            self.assertEqual(x.stride(), y.stride())

    @Dtypes(torch.half, torch.float)
    def test_full_inference(self, dtype):
        size = (2, 2)

        prev_default = torch.get_default_dtype()
        torch.set_default_dtype(dtype)

        # Tests bool fill value inference
        t = torch.full(size, True)
        self.assertEqual(t.dtype, torch.bool)

        # Tests integer fill value inference
        t = torch.full(size, 1)
        self.assertEqual(t.dtype, torch.long)

        # Tests float fill value inference
        t = torch.full(size, 1.)
        self.assertEqual(t.dtype, dtype)

        torch.set_default_dtype(prev_default)

    def test_full_out(self, device="npu"):
        size = (5,)
        output = torch.empty(size, device=device, dtype=torch.long)

        # verifies dtype/out conflict throws a RuntimeError
        with self.assertRaises(RuntimeError):
            torch.full(output.shape, 1., dtype=torch.float, out=output)

        # verifies out dtype overrides inference
        self.assertEqual(torch.full(output.shape, 1., out=output).dtype, output.dtype)
        self.assertEqual(torch.full(size, 1, out=output).dtype, output.dtype)

    def test_ones_like(self, device="npu"):
        expected = torch.ones(100, 100, device=device)

        res1 = torch.ones_like(expected)
        self.assertEqual(res1.to('cpu'), expected.to('cpu'))

        # test boolean tensor
        expected = torch.tensor([True, True], device=device, dtype=torch.bool)
        res1 = torch.ones_like(expected)
        self.assertEqual(res1.to('cpu'), expected.to('cpu'))

    def test_zeros_like(self, device="npu"):
        expected = torch.zeros((100, 100,), device=device)

        res1 = torch.zeros_like(expected)
        self.assertEqual(res1.to('cpu'), expected.to('cpu'))

    def test_scalar_tensortype(self):
        import numpy as np
        np.random.seed(1024)
        dtypes = [
            [np.bool_, [torch.npu.BoolTensor, torch.BoolTensor]],
            [np.float64, [torch.npu.DoubleTensor, torch.DoubleTensor]],
            [np.float32, [torch.npu.FloatTensor, torch.FloatTensor]],
            [np.float16, [torch.npu.HalfTensor, torch.HalfTensor]],
            [np.float32, [torch.npu.BFloat16Tensor, torch.BFloat16Tensor]],
            [np.int64, [torch.npu.LongTensor, torch.LongTensor]],
            [np.int32, [torch.npu.IntTensor, torch.IntTensor]],
            [np.int16, [torch.npu.ShortTensor, torch.ShortTensor]],
            [np.int8, [torch.npu.CharTensor, torch.CharTensor]],
            [np.uint8, [torch.npu.ByteTensor, torch.ByteTensor]],
        ]

        for item in dtypes:
            dt = item[0]
            tt = item[1]
            np_data = np.random.randn(2, 3, 4).astype(dt)
            npu_tensor = tt[0](np_data)
            cpu_tensor = tt[1](np_data)
            self.assertEqual(npu_tensor.dtype, cpu_tensor.dtype)
            self.assertEqual(npu_tensor.device.type, "npu")
            self.assertEqual(npu_tensor.to("cpu"), cpu_tensor)

    def test_faketensor(self, device="npu"):
        from torch.utils._mode_utils import no_dispatch
        from torch._subclasses.fake_tensor import FakeTensorMode
        faketensor_mode = FakeTensorMode(allow_non_fake_inputs=True)

        x = torch.randn((3, 3,), device=device)
        faketensor = faketensor_mode.from_tensor(x)

        expected = torch.zeros_like(x, device=x.device)
        with no_dispatch():
            res1 = torch.zeros_like(faketensor, device=faketensor.device)

        self.assertEqual(res1.to('cpu'), expected.to('cpu'))

    def test_empty_with_deterministic(self):
        with DeterministicGuard(True, fill_uninitialized_memory=True):
            empty_tensor = torch.empty(2, 3, 4)
            empty_strided_tensor = torch.empty_strided((2, 3, 4), (1, 1, 1))
            self.assertTrue(empty_tensor.isnan().all())
            self.assertTrue(empty_strided_tensor.isnan().all())


if __name__ == '__main__':
    run_tests()
