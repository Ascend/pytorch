# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
import unittest
import torch.nn as nn
import torch.nn .functional as F
import numpy as np
from torch._six import inf, nan, string_classes, istuple
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes
from util_test import create_common_tensor

class TestAddcmul(TestCase):
    def test_addcmul(self, device):
        def rand_tensor(size, dtype, device):
            if dtype.is_floating_point:
                #return torch.rand(size=size, dtype=dtype, device=device)
                return torch.rand(size=size, dtype=dtype, device="cpu").to(device)
            if dtype == torch.uint8:
                #return torch.randint(1, 5, size=size, dtype=dtype, device=device)
                return torch.randint(1, 5, size=size, dtype=dtype, device="cpu").to(device)
            else:
                #return torch.randint(-5, 5, size=size, dtype=dtype, device=device)
                return torch.randint(-5, 5, size=size, dtype=dtype, device="cpu").to(device)

        #for dtype in torch.testing.get_all_math_dtypes(device):
        for dtype in [torch.float32, torch.int]:
            a = rand_tensor((2, 2), dtype=dtype, device=device)
            b = rand_tensor((2, 2), dtype=dtype, device=device)
            c = rand_tensor((2, 2), dtype=dtype, device=device)
            if dtype.is_floating_point:
                alpha = 0.1
            else:
                alpha = 3
            actual = torch.addcmul(a, b, c, value=alpha)
            expected = a + alpha * b * c
            self.assertTrue(torch.allclose(expected.to("cpu"), actual.to("cpu")))

            with self.maybeWarnsRegex(
                    UserWarning, "This overload of addcmul is deprecated"):
                self.assertEqual(actual.to("cpu"), torch.addcmul(a, alpha, b, c).to("cpu"))

class TestThreshold(TestCase):
    #@onlyCPU
    #@dtypes(*torch.testing.get_all_math_dtypes('cpu'))
    def test_threshold(self, device):
        dtype = torch.float32
        #if dtype != torch.uint8 and dtype != torch.float16:
            # 100 is wide enough to use AVX2 instructions for all types
        #x = torch.randn(100, dtype=torch.float, device=device).sign().to(dtype=dtype)
        x = torch.randn(100, dtype=torch.float, device="cpu").sign().to(dtype=dtype).to(device)
        y = torch.threshold(x, 0, 0)
        self.assertTrue(y.le(0).any())

class TestNllloss(TestCase):
    def test_nll_loss_mismatched_batch(self, device):
        #x = torch.randn((10, 3), requires_grad=True, device=device)
        x = torch.randn((10, 3), requires_grad=True, device="cpu").to(device)
        # t should have size (10,)
        t = torch.zeros((3,), dtype=torch.int64, device=device)
        with self.assertRaisesRegex(ValueError, 'Expected.*batch_size'):
            F.nll_loss(x, t)

    def test_nll_loss_out_of_bounds_ignore_index(self, device):
        #x = torch.randn(6, 3, requires_grad=True, device=device)
        x = torch.randn(6, 3, requires_grad=True, device="cpu").to(device)
        #t = torch.tensor([0, 1, 255, 0, 1, 2], dtype=torch.int64, device=device)
        t = torch.tensor([0, 1, 255, 0, 1, 2], dtype=torch.int, device=device)
        for reduction in ['mean', 'none']:
            F.nll_loss(x, t, ignore_index=255, reduction=reduction).sum().backward()

    def _nll_loss_helper(self, input_size, reduction, expected, device):
        #input = torch.rand(input_size, requires_grad=True, device=device)
        input = torch.rand(input_size, requires_grad=True, device="cpu").to(device)
        num_channels = input_size[1]
        target_size = (input_size[0], ) + tuple(input_size[2:])
        #target = torch.randint(num_channels, target_size, device=device)
        target = torch.randint(num_channels, target_size, device="cpu").to(device)

        output = F.nll_loss(input, target, reduction=reduction)
        self.assertEqual(output, expected)

        output.sum().backward()
        self.assertEqual(input.grad.size(), input.size())

    @unittest.skip("unittest is unsuppoted")
    def test_nll_loss_empty_tensor_reduction_none(self, device):
        self._nll_loss_helper([0, 3], "none", torch.empty([0], device=device), device)
        self._nll_loss_helper([0, 3, 5, 7], "none", torch.empty([0, 5, 7], device=device), device)
        self._nll_loss_helper([2, 3, 0, 7], "none", torch.empty([2, 0, 7], device=device), device)
        self._nll_loss_helper([2, 3, 5, 0], "none", torch.empty([2, 5, 0], device=device), device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "none", torch.empty([2, 5, 7, 0], device=device), device)

    #@unittest.skipIf(TEST_WITH_UBSAN, "division-by-zero error with UBSAN")
    @unittest.skip("unittest is unsuppoted")
    def test_nll_loss_empty_tensor_reduction_mean(self, device):
        nan = torch.tensor(float('nan'), device=device)
        self._nll_loss_helper([0, 3], "mean", nan, device)
        self._nll_loss_helper([0, 3, 5, 7], "mean", nan, device)
        self._nll_loss_helper([2, 3, 0, 7], "mean", nan, device)
        self._nll_loss_helper([2, 3, 5, 0], "mean", nan, device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "mean", nan, device)

    @unittest.skip("unittest is unsuppoted")
    def test_nll_loss_empty_tensor_reduction_sum(self, device):
        zero = torch.tensor(0, device=device)
        self._nll_loss_helper([0, 3], "sum", zero, device)
        self._nll_loss_helper([0, 3, 5, 7], "sum", zero, device)
        self._nll_loss_helper([2, 3, 0, 7], "sum", zero, device)
        self._nll_loss_helper([2, 3, 5, 0], "sum", zero, device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "sum", zero, device)

    def test_nll_loss_total_weight_is_zero(self, device):

        def helper(input_size):
            input = torch.ones(input_size, requires_grad=True, device=device)
            num_channels = input_size[1]
            target_size = (input_size[0], ) + tuple(input_size[2:])
            #target = torch.zeros(target_size, dtype=torch.long, device=device)
            target = torch.zeros(target_size, dtype=torch.int, device=device)
            weight = torch.zeros([num_channels], device=device)
            self.assertEqual(F.nll_loss(input, target, weight).to("cpu").item(), 0)

        helper([2, 3])
        helper([2, 3, 5, 7])
        helper([2, 3, 5, 7, 9])

class TestLogsoftmax(TestCase):
    #@repeat_test_for_types([torch.float, torch.bfloat16])
    def test_log_softmax(self, device):
        dtype=torch.float
        x_small = torch.ones(1, 2, dtype=dtype)
        x_big = x_small + 1e16
        x_small = x_small.to("npu")
        x_big = x_big.to("npu")
        self.assertEqual(F.log_softmax(x_small, -1).to("cpu"), F.log_softmax(x_big, -1).to("cpu"),)
    
    @unittest.skip("unittest is unsuppoted")
    def test_log_softmax_cpu(self, device):
        dtype=torch.bfloat16
        inputf = torch.rand(32, 100, device="cpu", dtype=torch.float, requires_grad=True).to(device)
        input = inputf.to(dtype).detach().requires_grad_(True).to(device)
        outf = F.log_softmax(inputf, dim=-1)
        out = F.log_softmax(input, dim=-1)
        self.assertEqual(out.dtype, dtype)
        self.assertEqual(out.to("cpu"), outf.to("cpu"), prec=0.1)

        out.sum().backward()
        outf.sum().backward()
        self.assertEqual(input.grad.dtype, dtype)
        self.assertEqual(input.grad, inputf.grad.to(dtype), prec=0.1)

class TestOneslike(TestCase):
    def test_ones_like(self, device):
        expected = torch.ones(100, 100, device=device)

        res1 = torch.ones_like(expected)
        self.assertEqual(res1.to("cpu"), expected.to("cpu"))

    #@deviceCountAtLeast(2)
    def test_ones_like_multiple_device(self, device):
        devices=["npu:0", "npu:1"]
        expected = torch.ones(100, 100, device=devices[0])
        #x = torch.randn(100, 100, device=devices[1], dtype=torch.float32)
        x = torch.randn(100, 100, device="cpu", dtype=torch.float32).to(devices[1])
        output = torch.ones_like(x)
        self.assertEqual(output.to("cpu"), expected.to("cpu"))

    def test_ones_like_nn(self, device):
        expected = torch.ones(100, 100).to(device)

        res1 = torch.ones_like(expected)
        self.assertEqual(res1.to("cpu"), expected.to("cpu"))

        # test boolean tensor
        expected = torch.tensor([True, True], dtype=torch.bool).to(device)
        res1 = torch.ones_like(expected)
        self.assertEqual(res1.to("cpu"), expected.to("cpu"))

class TestOnes(TestCase):
    def test_ones(self, device):
        res1 = torch.ones(100, 100).to(device)
        #res2 = torch.Tensor()
        res2 = torch.Tensor(100, 100).to(device)
        torch.ones(100, 100, out=res2, device=device)
        #self.assertEqual(res1, res2)
        self.assertEqual(res1.to("cpu"), res2.to("cpu"))

        # test boolean tensor
        res1 = torch.ones(1, 2, dtype=torch.bool, device=device)
        expected = torch.tensor([[True, True]], dtype=torch.bool)
        self.assertEqual(res1.to("cpu"), expected.to("cpu"))

class TestSoftmax(TestCase):
    #@onlyCUDA
    #@skipCUDAIfRocm
    #@dtypes(torch.half, torch.float)
    @unittest.skip("unittest is unsuppoted")
    def test_softmax(self, device):
        dtype = torch.float
        #input = torch.rand(32, 100, device=device, dtype=dtype, requires_grad=True)
        input = torch.rand(32, 100, device="cpu", dtype=dtype, requires_grad=True).to(device)
        inputf = input.to(torch.float).detach().to(device).requires_grad_(True)
        out = F.softmax(input, dim=-1, dtype=torch.float)
        outf = F.softmax(inputf, dim=-1)
        # should be bitwise equal
        self.assertEqual(out.to("cpu"), outf.to("cpu"), prec=0)
        gO = torch.empty_like(outf).uniform_()
        out.backward(gO)
        outf.backward(gO)
        # should be bitwise equal
        self.assertEqual(input.grad.to("cpu"), inputf.grad.to(dtype).to("cpu"), prec=0)

    #@dtypesIfCUDA(torch.half, torch.float)
    #@dtypes(torch.float)
    def test_softmax_backward(self, device):
        dtype = torch.float
        #sizes = [(0, 10), (32, 20), (10, 0)]
        sizes = [(1, 10), (32, 20), (10, 1)]
        #for fn in [F.softmax, F.log_softmax]:
        for fn in [F.softmax]:
            for size in sizes:
                #input = torch.rand(size, device=device, dtype=dtype, requires_grad=True)
                input = torch.rand(size, device="cpu", dtype=dtype, requires_grad=True).to(device)
                for dim in [0, 1]:
                    output = fn(input, dtype=torch.float, dim=dim).sum()
                    grad_input, = torch.autograd.grad(output.to(device), input, create_graph=True)
                    grad_input.sum().backward()

class TestAddmm(TestCase):
    #@slowTest
    #@onlyCPU
    def test_addmm(self, device):
        types = {
            #'torch.DoubleTensor': 1e-8,
            'torch.FloatTensor': 1e-4,
            #'torch.BFloat16Tensor': 1e-1,
        }
        for tname, prec in types.items():
            #M = torch.randn(10, 25, device=device).type(tname)
            #m1 = torch.randn(10, 50, device=device).type(tname)
            #m2 = torch.randn(50, 25, device=device).type(tname)
            M = torch.randn(10, 25, device="cpu").type(tname).to(device)
            m1 = torch.randn(10, 50, device="cpu").type(tname).to(device)
            m2 = torch.randn(50, 25, device="cpu").type(tname).to(device)
            res1 = torch.addmm(M, m1, m2)
            #res2 = torch.zeros(10, 25, device=device).type(tname)
            res2 = torch.zeros(10, 25, device="cpu").type(tname).to(device)
            res2 += M
            for i in range(10):
                for j in range(25):
                    for k in range(50):
                        res2[i, j] += m1[i, k] * m2[k, j]
            self.assertEqual(res1.to("cpu"), res2.to("cpu"), prec)

        # Test 0-strided
        for tname, prec in types.items():
            #M = torch.randn(10, 1, device=device).type(tname).expand(10, 25)
            #m1 = torch.randn(10, 1, device=device).type(tname).expand(10, 50)
            #m2 = torch.randn(50, 25, device=device).type(tname)
            
            M = torch.randn(10, 1, device="cpu").type(tname).expand(10, 25).to(device)
            m1 = torch.randn(10, 1, device="cpu").type(tname).expand(10, 50).to(device)
            m2 = torch.randn(50, 25, device="cpu").type(tname).to(device)

            res1 = torch.addmm(M, m1, m2)
            #res2 = torch.zeros(10, 25, device=device).type(tname)
            res2 = torch.zeros(10, 25, device="cpu").type(tname).to(device)
            res2 += M
            for i in range(10):
                for j in range(25):
                    for k in range(50):
                        res2[i, j] += m1[i, k] * m2[k, j]
            self.assertEqual(res1.to("cpu"), res2.to("cpu"), prec)

    #@dtypes(torch.float, torch.double)
    def test_addmm_sizes(self, device):
        dtype=torch.float
        #for m in [0, 1, 25]:
            #for n in [0, 1, 10]:
                #for k in [0, 1, 8]:
        for m in [1, 1, 25]:
            for n in [1, 1, 10]:
                for k in [1, 1, 8]:
                    #M = torch.randn(n, m, device=device, dtype=dtype)
                    #m1 = torch.randn(n, k, device=device, dtype=dtype)
                    #m2 = torch.randn(k, m, device=device, dtype=dtype)
                    M = torch.randn(n, m, device="cpu", dtype=dtype).to(device)
                    m1 = torch.randn(n, k, device="cpu", dtype=dtype).to(device)
                    m2 = torch.randn(k, m, device="cpu", dtype=dtype).to(device)
                    res1 = torch.addmm(M, m1, m2)
                    res2 = torch.zeros(n, m, device=device, dtype=dtype)
                    res2 += M
                    for i in range(n):
                        for j in range(m):
                            for l in range(k):
                                res2[i, j] += m1[i, l] * m2[l, j]
                    self.assertEqual(res1.to("cpu"), res2.to("cpu"))

class Testmm(TestCase):
    #@slowTest
    #@onlyCPU
    def test_mm(self, device):
        def _test_mm(n, m, p, dtype, genf):
            # helper function
            def matrixmultiply(mat1, mat2):
                n = mat1.size(0)
                m = mat1.size(1)
                p = mat2.size(1)
                res = torch.zeros(n, p, dtype=dtype, device=device)
                for i, j in iter_indices(res):
                    res[i, j] = sum(mat1[i, k] * mat2[k, j] for k in range(m))
                return res

            # contiguous case
            mat1 = genf(n, m).to(device)
            mat2 = genf(m, p).to(device)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2).to(device)
            self.assertEqual(res.to("cpu"), res2.to("cpu"))

            # non contiguous case 1
            mat1 = genf(n, m).to(device)
            mat2 = genf(p, m).t().to(device)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2).to(device)
            self.assertEqual(res.to("cpu"), res2.to("cpu"))

            # non contiguous case 2
            mat1 = genf(m, n).t().to(device)
            mat2 = genf(m, p).to(device)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2).to(device)
            self.assertEqual(res.to("cpu"), res2.to("cpu"))

            # non contiguous case 3
            mat1 = genf(m, n).t().to(device)
            mat2 = genf(p, m).t().to(device)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2).to(device)
            self.assertEqual(res.to("cpu"), res2.to("cpu"))

            # test with zero stride
            mat1 = genf(n, m).to(device)
            mat2 = genf(m, 1).expand(m, p).to(device)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2).to(device)
            self.assertEqual(res.to("cpu"), res2.to("cpu"))

            # explicitly exercise the _out variant in torch.mm().
            # contiguous case
            mat1 = genf(n, m).to(device)
            mat2 = genf(m, p).to(device)
            res = genf(n, p).to(device)
            torch.mm(mat1, mat2, out=res)

            res2 = matrixmultiply(mat1, mat2).to(device)
            self.assertEqual(res.to("cpu"), res2.to("cpu"))

            # explicitly exercise the _out variant in torch.mm().
            # non contiguous case 3
            mat1 = genf(m, n).t().to(device)
            mat2 = genf(p, m).t().to(device)
            res = genf(n, p).to(device)
            torch.mm(mat1, mat2, out=res)

            res2 = matrixmultiply(mat1, mat2).to(device)
            self.assertEqual(res.to("cpu"), res2.to("cpu"))

        #for (n, m, p) in [(20, 10, 5), (15, 5, 10), (5, 18, 10)]:
            #_test_mm(n, m, p, torch.float32, lambda x, y: torch.randn(x, y, dtype=torch.float32, device=device))
            #_test_mm(n, m, p, torch.float64, lambda x, y: torch.randn(x, y, dtype=torch.float64, device=device))
            #_test_mm(n, m, p, torch.int32, lambda x, y: torch.randint(0, 100, (x, y), dtype=torch.int32, device=device))
            #_test_mm(n, m, p, torch.int64, lambda x, y: torch.randint(0, 100, (x, y), dtype=torch.int64, device=device))
            #_test_mm(n, m, p, torch.bfloat16, lambda x, y: torch.randn(x, y, dtype=torch.float32, device=device).bfloat16())
        for (n, m, p) in [(20, 10, 5), (15, 5, 10), (5, 18, 10)]:
            _test_mm(n, m, p, torch.float32, lambda x, y: torch.randn(x, y, dtype=torch.float32, device="cpu"))
            #_test_mm(n, m, p, torch.float64, lambda x, y: torch.randn(x, y, dtype=torch.float64, device="cpu"))
            _test_mm(n, m, p, torch.int32, lambda x, y: torch.randint(0, 100, (x, y), dtype=torch.int32, device="cpu"))
            #_test_mm(n, m, p, torch.int64, lambda x, y: torch.randint(0, 100, (x, y), dtype=torch.int64, device="cpu"))
            #_test_mm(n, m, p, torch.bfloat16, lambda x, y: torch.randn(x, y, dtype=torch.float32, device="cpu").bfloat16())

class TestTranspose(TestCase):
    def is_view_of(self, base, other):
        if (not other._is_view() or
                other is base or
                other._base is not base or
                base.device != other.device):
            return False

        # Note: only validates storage on native device types
        # because some accelerators, like XLA, do not expose storage
        #if base.device.type == 'cpu' or base.device.type == 'npu':
            #if base.storage().data_ptr() != other.storage().data_ptr():
               # return False

        return True

    def test_transpose_view(self, device):
        t = torch.ones((5, 5), device=device)
        v = torch.transpose(t, 0, 1)
        self.assertTrue(self.is_view_of(t, v))

        #v[0, 1] = 0
        #self.assertEqual(t[1, 0].to("cpu"), v[0, 1].to("cpu"))

class TestT(TestCase):
    @unittest.skip("unittest is unsuppoted")
    def test_t(self):
        # Test 0D tensors
        x = torch.randn(())
        self.assertEqual(x, x.t())
        x = x.to_sparse()
        self.assertEqual(x, x.t())

        # Test 1D tensors
        x = torch.arange(4)
        self.assertEqual(x, x.t())
        x = x.to_sparse()
        self.assertEqual(x, x.t())

        # Test 2D tensors
        x = torch.rand((2, 2))
        self.assertEqual(x.t(), x.transpose(0, 1))
        x = x.to_sparse()
        self.assertEqual(x.t(), x.transpose(0, 1))

        # Test 3D tensor
        x = torch.rand((2, 2, 2))
        with self.assertRaisesRegex(RuntimeError, 'expects a tensor with <= 2 dimensions, but self is 3D'):
            x.t()
        x = x.to_sparse()
        with self.assertRaisesRegex(RuntimeError, 'expects a tensor with <= 2 sparse and 0 dense dimensions'):
            x.t()

class TestClone(TestCase):
    def test_clone_all_dtypes_and_devices(self, device):
        #for dt in torch.testing.get_all_dtypes():
        for dt in [torch.float, torch.int, torch.float16]:
            x = torch.tensor((1, 1), dtype=dt, device=device)
            y = x.clone()
            self.assertEqual(x.to("cpu"), y.to("cpu"))

    def test_clone_zero_stride_dim(self, device):
        # stride zero, size 1 axis, not contiguous
        x = torch.randn(10).to(device)
        y = x.as_strided([2, 1, 5], [1, 0, 2])
        self.assertEqual(y.to("cpu"), y.clone().to("cpu"))

class TestSqrt(TestCase):
    def _test_math(self, torchfn, mathfn, input=None, test_expand=False, rtol=None, atol=None):
        device = "npu:0"
        if input is None:
            input = []
            input.append(list(range(-5, 5)))
            input.append([0 for x in range(-5, 5)])
            input.append([x + 1e-6 for x in range(-5, 5)])
            # Some vectorized implementations don't support large ranges
            input.append([x + 1e10 for x in range(-5, 5)])
            input.append([x - 1e10 for x in range(-5, 5)])
            input.append(torch.randn(10).tolist())
            input.append((torch.randn(10) + 1e6).tolist())
            input.append([math.pi * (x / 2) for x in range(-5, 5)])

        def compare_reference(input, dtype):
            #input = torch.tensor(input, dtype=dtype)
            input = torch.tensor(input, dtype=dtype)
            res1 = torchfn(input.clone())
            res2 = input.clone().apply_(mathfn)
            torch.testing.assert_allclose(res1.to("cpu"), res2.to("cpu"), rtol=rtol, atol=atol)

        # compare against the reference math function
        #compare_reference(input, torch.double)
        #compare_reference(input, torch.float, device)

        def check_non_contiguous(shape, dtype):
            contig = torch.randn(shape, dtype=dtype)
            non_contig = torch.empty(shape + (2,), dtype=dtype, device=device)[..., 0]
            non_contig.copy_(contig)
            self.assertFalse(non_contig.is_contiguous())
            self.assertEqual(torchfn(contig), torchfn(non_contig).to("cpu"), 'non-contiguous')

        # compare application against contiguous vs. non-contiguous
        #check_non_contiguous((5, 7), torch.double)
        #check_non_contiguous((1024,), torch.double)
        #check_non_contiguous((5, 7), torch.float)
        #check_non_contiguous((1024,), torch.float)

        def check_non_contiguous_index(dtype):
            contig = torch.randn((2, 2, 1, 2), dtype=dtype)
            non_contig = contig[:, 1, ...]
            contig = non_contig.clone().to(device)
            self.assertFalse(non_contig.is_contiguous())
            self.assertEqual(torchfn(contig).to("cpu"), torchfn(non_contig), 'non-contiguous index')

        #check_non_contiguous_index(torch.float)
        #check_non_contiguous_index(torch.double)

        def check_non_contiguous_expand(shape, dtype):
            contig = torch.randn(shape, dtype=dtype)
            non_contig = contig.clone().to(device).expand(3, -1, -1)
            self.assertFalse(non_contig.is_contiguous())
            contig = torchfn(contig)
            non_contig = torchfn(non_contig)
            for i in range(3):
                self.assertEqual(contig, non_contig[i].to("cpu"), 'non-contiguous expand[' + str(i) + ']')

        # Expand is not defined for in-place operations
        #if test_expand:
            # The size 1 case is special as it leads to 0 stride and needs to persists
            #check_non_contiguous_expand((1, 3), torch.double)
            #check_non_contiguous_expand((1, 7), torch.double)
            #check_non_contiguous_expand((5, 7), torch.float)

        # If size(dim) == 1, stride(dim) is not defined.
        # The code needs to be able to handle this
        def check_contiguous_size1(dtype):
            contig = torch.randn((5, 100), dtype=dtype)
            contig = contig[:1, :50]
            contig2 = torch.empty(contig.size(), dtype=dtype, device=device)
            contig2.copy_(contig)
            self.assertTrue(contig.is_contiguous())
            self.assertTrue(contig2.is_contiguous())
            self.assertEqual(torchfn(contig), torchfn(contig2).to("cpu"), 'contiguous size1')

        #check_contiguous_size1(torch.double)
        #check_contiguous_size1(torch.float)

        def check_contiguous_size1_largedim(dtype):
            contig = torch.randn((5, 2, 3, 1, 4, 5, 3, 2, 1, 2, 3, 4), dtype=dtype)
            contig = contig[:1, :, :, :, :, :, :, :, :, :, :, :]
            contig2 = torch.empty(contig.size(), dtype=dtype, device=device)
            contig2.copy_(contig)
            self.assertTrue(contig.is_contiguous())
            self.assertTrue(contig2.is_contiguous())
            self.assertEqual(torchfn(contig), torchfn(contig2).to("cpu"), 'contiguous size1')

        #check_contiguous_size1_largedim(torch.double)
        #check_contiguous_size1_largedim(torch.float)

        def check_large(dtype):
            input = torch.randn(1024, 512, dtype=dtype).to(device)
            actual = torchfn(input)
            expected = torch.stack([torchfn(slice) for slice in input])
            self.assertEqual(actual.to("cpu"), expected, 'large')

        # compare large tensor vs. repeated small applications to expose
        # possible parallelism bugs.
        #check_large(torch.double)
        check_large(torch.float)
    
    def test_sqrt(self, device):
        self._test_math(torch.sqrt, lambda x: math.sqrt(x) if x >= 0 else nan)

class TestResize(TestCase):
    def test_resize_all_dtypes_and_devices(self, device):
        shape = (2, 2)
        #for dt in torch.testing.get_all_dtypes():
        for dt in [torch.float, torch.int, torch.float16]:
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            x.resize_(shape)
            self.assertEqual(shape, x.shape)

    def test_resize_as_all_dtypes_and_devices(self, device):
        #for dt in torch.testing.get_all_dtypes():
        for dt in [torch.float, torch.int, torch.float16]:
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            y = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dt, device=device)
            x.resize_as_(y)
            self.assertEqual(y.shape, x.shape)

    def test_resize_as_preserves_strides(self, device):
        x = torch.empty(2, 3).t().to(device)
        old_strides = x.stride()
        x.resize_as_(x)
        self.assertEqual(x.stride(), old_strides)

class TestConv2d(TestCase):
    @unittest.skip("unittest is unsupported")
    def test_invalid_conv2d(self, device):
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            module = torch.nn.Conv2d(1, 1, kernel_size=3, dilation=2, stride=2).to(dtype).to("npu")
            input = torch.empty(1, 1, 4, 4).to(dtype).to("npu")
            self.assertRaises(RuntimeError, lambda: module(input))

            module = nn.Conv2d(in_channels=3, out_channels=33, kernel_size=10, stride=1, bias=True).to("npu")
            input = torch.randn(1, 3, 1, 1).to("npu")
            with self.assertRaisesRegex(RuntimeError,
                                        r'Calculated padded input size per channel: \(1 x 1\). ' +
                                        r'Kernel size: \(10 x 10\). Kernel size can\'t be greater than actual input size'):
                module(input)

            # Negative stride check
            module = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=-1, bias=True).to(dtype).to("npu")
            input = torch.randn(1, 3, 4, 4).to(dtype).to("npu")
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

            # Zero stride check
            module = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=0, bias=True).to(dtype).to("npu")
            input = torch.randn(1, 3, 4, 4).to(dtype).to("npu")
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

    @unittest.skip("unittest is unsupported")
    def test_Conv2d_deterministic(self, device):
        dtype=torch.float
        inputs = torch.randn(2, 3, 5, 5, device="cpu", dtype=dtype, requires_grad=True).to("npu")
        #with cudnn.flags(enabled=True, benchmark=True, deterministic=True):
        
        conv1 = torch.nn.Conv2d(3, 3, 3).to("npu", dtype)
        conv2 = torch.nn.Conv2d(3, 3, 3).to("npu", dtype)
        conv2.bias.data.copy_(conv1.bias.data)
        conv2.weight.data.copy_(conv1.weight.data)
        out1 = conv1(inputs)
        out2 = conv2(inputs)
        self.assertEqual(out1.to("cpu"), out2.to("cpu"), prec=0.0)
        y = torch.randn(out1.size(), device="cpu", dtype=dtype).to("npu")
        out1.backward(y)
        out2.backward(y)
        self.assertEqual(conv1.bias.grad.data.to("cpu"), conv2.bias.grad.data.to("cpu"), prec=0.0)
        self.assertEqual(conv1.weight.grad.data.to("cpu"), conv2.weight.grad.data.to("cpu"), prec=0.0)

    def test_Conv2d_missing_argument(self, device):
        c = nn.Conv2d(3, 3, 3).to("npu")
        self.assertRaises(TypeError, lambda: c(None).to("cpu"))

    def test_Conv2d_backward_twice(self, device):
        input = torch.randn(2, 3, 5, 5).to("npu")
        c = nn.Conv2d(3, 3, 3).to("npu")
        o1 = c(input)
        o1.sum().backward()
        self.assertRaisesRegex(RuntimeError, 'Specify retain_graph=True',
                               lambda: o1.sum().backward().to("cpu"))

    def test_Conv2d_large_workspace(self, device):
        dtype=torch.float
        # These sizes require huge cuDNN workspaces. Make sure we choose a
        # reasonable algorithm that does not run out of memory
        sizes = [
            (1, 256, 109, 175),
            (1, 256, 80, 128),
            (1, 256, 120, 192),
        ]

        def run_test(benchmark):
            #with torch.backends.cudnn.flags(benchmark=benchmark):
            conv = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1).to("npu", dtype)
            for size in sizes:
                x = torch.randn(size, device="cpu", dtype=dtype).to("npu")
                out = conv(x.detach().clone().requires_grad_())
                out.backward(torch.ones_like(out))

        run_test(benchmark=False)
        run_test(benchmark=True)

    # For https://github.com/pytorch/pytorch/pull/1273
    # Almost identical to the above `test_Conv2d_naive_groups`
    @unittest.skip("unittest is unsupported")
    def test_Conv2d_groups_nobias(self, device):
        dev_dtypes += [("npu", torch.float), ("npu", torch.half)]
        for device, dtype in dev_dtypes:
            m = nn.Conv2d(4, 4, kernel_size=3, groups=2, bias=False).to(device, dtype)
            i = torch.randn(2, 4, 6, 6, device="cpu", dtype=dtype, requires_grad=True).to(device)
            output = m(i)
            grad_output = torch.randn(2, 4, 4, 4, device="cpu", dtype=dtype).to(device)
            output.backward(grad_output)

            m1 = nn.Conv2d(2, 2, kernel_size=3, bias=False).to(device, dtype)
            m1.weight.data.copy_(m.weight.data[:2])
            i1 = i.data[:, :2].contiguous().requires_grad_(True)
            output1 = m1(i1)
            output1.backward(grad_output[:, :2].contiguous())

            m2 = nn.Conv2d(2, 2, kernel_size=3, bias=False).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[2:])
            i2 = i.data[:, 2:].contiguous().requires_grad_(True)
            output2 = m2(i2)
            output2.backward(grad_output[:, 2:].contiguous())

            self.assertEqual(output.to("cpu"), torch.cat([output1, output2], 1).to("cpu"))
            self.assertEqual(i.grad.data.to("cpu"),
                             torch.cat([i1.grad.data, i2.grad.data], 1).to("cpu"),
                             dtype2prec_DONTUSE[dtype])
            self.assertEqual(m.weight.grad.data.to("cpu"),
                             torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0).to("cpu"),
                             1e-1 if dtype == torch.half else dtype2prec_DONTUSE[dtype])

    # Almost identical to the above `test_Conv2d_naive_groups`
    # Covering special case when group > 1, input-channel / group < 16 and output-channel is multiple of 16
    # See also https://github.com/pytorch/pytorch/pull/18463#issuecomment-476563686
    # and https://github.com/pytorch/pytorch/pull/18463#issuecomment-477001024
    @unittest.skip("unittest is unsupported")
    def test_Conv2d_groups_nobias_v2(self, device):
        torch.manual_seed(123)
        dev_dtypes += [("npu", torch.float), ("npu", torch.half)]
        for device, dtype in dev_dtypes:
            m = nn.Conv2d(4, 16, kernel_size=3, groups=2, bias=False).to(device, dtype)
            i = torch.randn(2, 4, 6, 6, device="cpu", dtype=dtype, requires_grad=True).to(device)
            output = m(i)
            grad_output = torch.randn(2, 16, 4, 4, device="cpu", dtype=dtype).to(device)
            output.backward(grad_output)

            m1 = nn.Conv2d(2, 8, kernel_size=3, bias=False).to(device, dtype)
            m1.weight.data.copy_(m.weight.data[:8])
            i1 = i.data[:, :2].contiguous().requires_grad_(True)
            output1 = m1(i1)
            output1.backward(grad_output[:, :8].contiguous())

            m2 = nn.Conv2d(2, 8, kernel_size=3, bias=False).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[8:])
            i2 = i.data[:, 2:].contiguous().requires_grad_(True)
            output2 = m2(i2)
            output2.backward(grad_output[:, 8:].contiguous())

            self.assertEqual(output.to("cpu"), torch.cat([output1, output2], 1).to("cpu"))
            self.assertEqual(i.grad.data.to("cpu"),
                             torch.cat([i1.grad.data, i2.grad.data], 1).to("cpu"),
                             dtype2prec_DONTUSE[dtype])
            self.assertEqual(m.weight.grad.data.to("cpu"),
                             torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0).to("cpu"),
                             1e-1 if dtype == torch.half else dtype2prec_DONTUSE[dtype])
    
    # Very similar to test_Conv2d_naive_groups but with special care to handle
    # the number of groups == number of input channels
    @unittest.skip("unittest is unsupported")
    def test_Conv2d_depthwise_naive_groups(self, device):
        dtype=torch.float
        for depth_multiplier in [1, 2]:
            m = nn.Conv2d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to("npu", dtype)
            i = torch.randn(2, 2, 6, 6, device="cpu", dtype=dtype).div_(2).requires_grad_().to("npu")
            output = m(i)
            grad_output = torch.randn(2, 2 * depth_multiplier, 4, 4, device="cpu", dtype=dtype) / 2
            grad_output = grad_output.to("npu")
            output.backward(grad_output)

            offset = 1 * depth_multiplier

            m1 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to("npu", dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = i.detach()[:, :1].clone().requires_grad_()
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())

            m2 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to("npu", dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())

            self.assertEqual(output.to("cpu"), torch.cat([output1, output2], 1).to("cpu"),
                             prec=dtype2prec_DONTUSE[dtype])
            self.assertEqual(i.grad.data.to("cpu"),
                             torch.cat([i1.grad.data, i2.grad.data], 1).to("cpu"),
                             prec=dtype2prec_DONTUSE[dtype])
            self.assertEqual(m.bias.grad.data.to("cpu"),
                             torch.cat([m1.bias.grad.data,
                                        m2.bias.grad.data], 0).to("cpu"),
                             prec=dtype2prec_DONTUSE[dtype])
            self.assertEqual(m.weight.grad.data.to("cpu"),
                             torch.cat([m1.weight.grad.data, 
                                        m2.weight.grad.data], 0).to("cpu"),
                             prec=dtype2prec_DONTUSE[dtype])

class Testto(TestCase):
    @unittest.skip("unittest is unsupported")
    def test_to(self, device):
        def test_copy_behavior(t, non_blocking=False):
            self.assertIs(t, t.to(t, non_blocking=non_blocking))
            self.assertIs(t, t.to(t.dtype, non_blocking=non_blocking))
            self.assertIs(t, t.to(torch.empty_like(t), non_blocking=non_blocking))
            self.assertIsNot(t, t.to(t, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(t.dtype, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(torch.empty_like(t), non_blocking=non_blocking, copy=True))

            #devices = [t.device]
            #if t.device.type == 'npu':
                #if t.device.index == -1:
                  #  devices.append('cuda:{}'.format(torch.cuda.current_device()))
                #elif t.device.index == torch.cuda.current_device():
                    #devices.append('npu')
            devices = ['npu']
            for device in devices:
                self.assertIs(t, t.to(device, non_blocking=non_blocking))
                self.assertIs(t, t.to(device, t.dtype, non_blocking=non_blocking))
                self.assertIsNot(t, t.to(device, non_blocking=non_blocking, copy=True))
                self.assertIsNot(t, t.to(device, t.dtype, non_blocking=non_blocking, copy=True))

        a = torch.tensor(5)
        test_copy_behavior(a)
        self.assertEqual(a.device, a.to('cpu').device)
        self.assertEqual(a.device, a.to('cpu', dtype=torch.float32).device)
        self.assertIs(torch.float32, a.to('cpu', dtype=torch.float32).dtype)
        self.assertEqual(a.device, a.to(torch.float32).device)
        self.assertIs(torch.float32, a.to(dtype=torch.float32).dtype)
        self.assertEqual(a.data_ptr(), a.to('cpu').data_ptr())
        self.assertEqual(a.data_ptr(), a.to(dtype=a.dtype, device=a.device, copy=False).data_ptr())
        self.assertEqual(a.data_ptr(), a.to('cpu', copy=False).data_ptr())
        self.assertNotEqual(a.data_ptr(), a.to('cpu', copy=True).data_ptr())

        #if torch.cuda.is_available():
        for non_blocking in [True, False]:
            #for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
            b = torch.tensor(5., device='npu')
            test_copy_behavior(b, non_blocking)
            self.assertEqual(b.device, b.to('npu', non_blocking=non_blocking).device)
            self.assertEqual(a.device, b.to('cpu', non_blocking=non_blocking).device)
            self.assertEqual(b.device, a.to('npu', non_blocking=non_blocking).device)
            self.assertIs(torch.int32, b.to('cpu', dtype=torch.int32, non_blocking=non_blocking).dtype)
            self.assertEqual(a.device, b.to('cpu', dtype=torch.int32, non_blocking=non_blocking).device)
            self.assertIs(torch.int32, b.to(dtype=torch.int32).dtype)
            self.assertEqual(b.device, b.to(dtype=torch.int32).device)

    def test_to_with_tensor(self, device):
        a = torch.tensor(5)
        self.assertEqual(a.device, a.to(a).device)

        #if torch.cuda.is_available():
        for non_blocking in [True, False]:
            #for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
            b = torch.tensor(5., device='npu')
            self.assertEqual(b.device, b.to(b, non_blocking=non_blocking).device)
            self.assertEqual(a.device, b.to(a, non_blocking=non_blocking).device)
            self.assertEqual(b.device, a.to(b, non_blocking=non_blocking).device)

    @unittest.skip("unittest is unsupported")
    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_float_to_int_undefined_conversion(self, device, dtype):
        t = torch.tensor((-3.40282e+38, 3.40282e+38), device=device, dtype=torch.float)
        self.assertEqual(t.to(dtype).dtype, dtype)

    def test_to_memory_format(self, device):
        m = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, bias=True)
        m = m.to(memory_format=torch.channels_last)
        for param in m.parameters():
            if param.dim() == 4:
                self.assertTrue(param.is_contiguous(memory_format=torch.channels_last))
              
class TestMaxPool2d(TestCase):

    #@onlyCUDA
    #@dtypesIfCUDA(torch.half, torch.float, torch.double)
    def test_max_pool2d_nhwc(self, device, dtype):
        def helper(n, c, h, w, kernel_size, stride=None):
            if stride is None:
                stride = kernel_size

            #input = torch.randn(n, c, h, w, dtype=dtype, device=device)
            input = torch.randn(n, c, h, w, dtype=dtype, device='cpu')
            input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
            grad = torch.randn(n, c, (h - kernel_size) // stride + 1, (w - kernel_size) // stride + 1,
                               dtype=dtype, device=device)
            pool = torch.nn.MaxPool2d(kernel_size, stride).to(device)

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.MaxPool2d(kernel_size, stride).to(device)

            out = pool(input.to(device))
            out.backward(grad.to(device))
            ref_out = ref_pool(ref_input.to(device))
            ref_out.backward(ref_grad.to(device))

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(torch.allclose(out.to(device), ref_out.to(device)))
            self.assertTrue(torch.allclose(input.grad.to(device), ref_input.grad.to(device)))

        helper(4, 8, 8, 8, 7)
        helper(200, 512, 28, 28, 2)
        helper(4, 8, 7, 7, 3, stride=1)
        helper(10, 512, 31, 31, 3, stride=2)
        helper(1, 129, 8, 8, 3, stride=2)
    

    def _test_maxpool_indices(self, num_dim, adaptive=False, device="npu", dtype=torch.float):
        def expected_indices(dim):
            if dim == 1:
                return torch.tensor([1, 3], dtype=torch.double).repeat(2, 2, 1)
            if dim == 2:
                return torch.tensor([[5, 7], [13, 15]], dtype=torch.double).repeat(2, 2, 1, 1)

        def expected_grad(dim):
            if dim == 1:
                return torch.tensor([0, 1, 0, 1], dtype=torch.double).repeat(2, 2, 1)
            grad = expected_grad(dim - 1)
            zero = torch.zeros(grad.size())
            return torch.stack((zero, grad, zero, grad), 2)

        def expected_output(dim):
            if dim == 1:
                return torch.arange(2, 17, 2).view(2, 2, 2)
            if dim == 2:
                col = torch.arange(6, 63, 8)
                return torch.stack([col, col + 2], 1).view(2, 2, 2, 2)

        if adaptive:
            cls_name = 'AdaptiveMaxPool{}d'.format(num_dim)
        else:
            cls_name = 'MaxPool{}d'.format(num_dim)
        module_cls = getattr(nn, cls_name)
        module = module_cls(2, return_indices=True).to(device, dtype=dtype)
        numel = 4 ** (num_dim + 1)
        input = torch.arange(1, numel + 1).view(2, 2, *repeat(4, num_dim)).to(device, dtype=dtype)
        input_var = input.clone().detach().requires_grad_()

        # Check forward
        output, indices = module(input_var)
        if num_dim != 3:
            expected_indices = expected_indices(num_dim)
            expected_output = expected_output(num_dim)
            self.assertEqual(indices.dim(), input.dim())
            self.assertEqual(indices.data.squeeze(), expected_indices)
            self.assertEqual(output.data.squeeze(), expected_output)
        self.assertTrue(output.requires_grad)
        self.assertFalse(indices.requires_grad)

        # Make sure backward works
        grad_output = torch.ones(output.size(), device=device, dtype=dtype)
        output.backward(grad_output, retain_graph=True)
        expected_grad = expected_grad(num_dim)
        self.assertEqual(input_var.grad.data, expected_grad.view_as(input))

        # Make sure backward after changing indices will result in an error
        indices.add_(1)
        self.assertRaises(RuntimeError, lambda: output.backward(grad_output))

        # Make sure -Infinity is handled correctly
        t = torch.tensor([[[float("-inf")]]])
        m = nn.MaxPool1d(kernel_size=1, return_indices=True)
        output, indices = m(t)
        self.assertEqual(output[0, 0, 0], float("-inf"), allow_inf=True)
        self.assertEqual(indices[0, 0, 0], 0)

        t = torch.tensor([[[float("-inf")]]])
        m = nn.MaxPool2d(kernel_size=1, return_indices=True)
        output, indices = m(t)
        self.assertEqual(output[0, 0, 0], float("-inf"), allow_inf=True)
        self.assertEqual(indices[0, 0, 0], 0)

        t = torch.tensor([[[[float("-inf")]]]])
        m = nn.MaxPool3d(kernel_size=1, return_indices=True)
        output, indices = m(t)
        self.assertEqual(output[0, 0, 0, 0], float("-inf"), allow_inf=True)
        self.assertEqual(indices[0, 0, 0, 0], 0)
    
    #@dtypesIfCUDA(*ALL_TENSORTYPES2)
    @dtypes(torch.float)
    def test_MaxPool2d_indices(self, device, dtype):
        self._test_maxpool_indices(2, device=device, dtype=dtype)

class TestSet(TestCase):
    def test_tensor_set(self, device):
        t1 = torch.Tensor().to(device)
        t2 = torch.Tensor(3, 4, 9, 10).uniform_().to(device)
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
        t1 = torch.Tensor()
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

        t1 = torch.tensor([True, True], dtype=torch.bool)
        t2 = torch.tensor([False, False], dtype=torch.bool)
        t1.set_(t2)
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)

    def test_tensor_set_errors(self, device):
        f_cpu = torch.randn((2, 3), dtype=torch.float32).to(device)
        d_cpu = torch.randn((2, 3), dtype=torch.float64).to(device)

        # change dtype
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(d_cpu.storage()))
        self.assertRaises(RuntimeError,
                          lambda: f_cpu.set_(d_cpu.storage(), 0, d_cpu.size(), d_cpu.stride()))
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(d_cpu))

        # change device
        #if torch.cuda.is_available():
        #f_cuda = torch.randn((2, 3), dtype=torch.float32, device='cuda')
        f_npu = torch.randn((2, 3), dtype=torch.float32, device='cpu').to(device)
        # cpu -> cuda
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(f_npu.storage()))
        self.assertRaises(RuntimeError,
                              lambda: f_cpu.set_(f_npu.storage(), 0, f_npu.size(), f_npu.stride()))
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(f_npu))

        # cuda -> cpu
        self.assertRaises(RuntimeError, lambda: f_npu.set_(f_cpu.storage()))
        self.assertRaises(RuntimeError,
                              lambda: f_npu.set_(f_cpu.storage(), 0, f_cpu.size(), f_cpu.stride()))
        self.assertRaises(RuntimeError, lambda: f_npu.set_(f_cpu))

class TestView(TestCase):
    def test_view_empty(self, device):
        x = torch.randn(0, 6).to("npu")
        self.assertEqual((1, 0, 6, 1, 1), x.view(1, 0, 6, 1, 1).shape)

    @unittest.skip("unittest is unsupported")
    def test_view(self, device):
        tensor = torch.rand(15, device=device).to("npu")
        template = torch.rand(3, 5, device=device).to("npu")
        empty = torch.empty(0, device=device)
        target = template.size()
        self.assertEqual(tensor.view_as(template).size(), target)
        self.assertEqual(tensor.view(3, 5).size(), target)
        self.assertEqual(tensor.view(torch.Size([3, 5])).size(), target)
        self.assertEqual(tensor.view(-1, 5).size(), target)
        self.assertEqual(tensor.view(3, -1).size(), target)
        tensor_view = tensor.view(5, 3)
        tensor_view.fill_(random.uniform(0, 1))
        self.assertEqual(empty.view_as(empty), empty)
        self.assertEqual(empty.view(0), empty)
        self.assertEqual(empty.view(0, 3, 0, 1).size(), torch.Size([0, 3, 0, 1]))
        self.assertEqual(empty.view(0, 3, 0, 1).view(0), empty)

        # test size inference with empty tensors
        self.assertEqual(empty.view(-1).size(), torch.Size([0]))
        self.assertEqual(empty.view(10, 3, -1).size(), torch.Size([10, 3, 0]))

        with self.assertRaisesRegex(RuntimeError, r"because the unspecified dimension size -1 can be any value"):
            empty.view(-1, 0)

        with self.assertRaisesRegex(RuntimeError, r"because the unspecified dimension size -1 can be any value"):
            empty.view(3, 0, -1, 0)

        self.assertRaises(RuntimeError, lambda: tensor.view(15, 0))
        self.assertRaises(RuntimeError, lambda: tensor.view(7, -1))
        self.assertRaises(RuntimeError, lambda: tensor.view(15, -1, -1))

        # test view when tensor is not contiguous in every dimension, but only
        # contiguous dimensions are touched.
        tensor = torch.rand(4, 2, 5, 1, 6, 2, 9, 3, device=device).transpose(-1, 2).transpose(-2, 3)
        # size:                      [   4,    2,    3,    9,    6,    2,    1,    5]
        # stride:                    [3840, 1620,    1,    3,   54,   27,  324,  324]
        # contiguous dim chunks:     [__________, ____, ____, __________, ____, ____]
        # merging 1 to chunk after:  [__________, ____, ____, __________, __________]
        contig_tensor = tensor.clone()
        # [4, 2] => [8, 1]
        # [3] => [3]
        # [9] => [3, 3]
        # [6, 2] => [4, 1, 3]
        # [1, 5] => [5]
        view_size = [8, 1, 3, 3, 3, 4, 1, 3, 5]
        self.assertEqual(tensor.view(*view_size), contig_tensor.view(*view_size))
        # [4, 2] => [2, 4]
        # [3] => [3]
        # [9] => [1, 9]
        # [6, 2] => [2, 2, 3]
        # [1, 5] => [5, 1]
        view_size = [2, 4, 3, 1, 9, 2, 2, 3, 5, 1]
        self.assertEqual(tensor.view(*view_size), contig_tensor.view(*view_size))
        # adding size 1 dims
        view_size = [1, 1, 2, 1, 4, 3, 1, 1, 9, 1, 2, 1, 2, 3, 1, 5, 1, 1]
        self.assertEqual(tensor.view(*view_size), contig_tensor.view(*view_size))

        # invalid views
        self.assertRaises(RuntimeError, lambda: tensor.view(-1))
        # crossing [4, 2], [3]
        self.assertRaises(RuntimeError, lambda: tensor.view(24, 9, 6, 2, 1, 5))
        # crossing [6, 2], [1, 5]
        self.assertRaises(RuntimeError, lambda: tensor.view(8, 3, 9, 6, 10))
        # crossing [9], [6, 2]
        self.assertRaises(RuntimeError, lambda: tensor.view(8, 3, 54, 2, 1, 5))

        # view with stride 0 dims
        tensor = torch.empty(1, 1, device=device).expand(3, 4)  # all dims are contiguous
        contig_tensor = tensor.clone()
        self.assertEqual(tensor.view(-1), contig_tensor.view(-1))
        self.assertEqual(tensor.view(1, -1, 1), contig_tensor.view(1, -1, 1))
        self.assertEqual(tensor.view(-1, 1), contig_tensor.view(-1, 1))
        self.assertEqual(tensor.view(6, 2, 1), contig_tensor.view(6, 2, 1))
        self.assertEqual(tensor.view(1, 6, 2, 1), contig_tensor.view(1, 6, 2, 1))
    
    def test_view_all_dtypes_and_devices(self, device):
        for dt in torch.testing.get_all_dtypes():
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            self.assertEqual(x.view(6).shape, [6])

    @unittest.skip("unittest is unsupported")
    def test_view_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t.view(25).to("npu")
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_view_as_view(self, device):
        t = torch.ones(5, 5, device=device)
        e = torch.empty((25,)).to("npu")
        v = t.view_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

def do_test_empty_full(self, dtypes, layout, device):
    shape = torch.Size([2, 3])

    def check_value(tensor, dtype, layout, device, value, requires_grad):
        self.assertEqual(shape, tensor.shape)
        self.assertIs(dtype, tensor.dtype)
        self.assertIs(layout, tensor.layout)
        self.assertEqual(tensor.requires_grad, requires_grad)
        if tensor.is_cuda and device is not None:
            self.assertEqual(device, tensor.device)
        if value is not None:
            fill = tensor.new(shape).fill_(value)
            self.assertEqual(tensor, fill)

    def get_int64_dtype(dtype):
        module = '.'.join(str(dtype).split('.')[1:-1])
        if not module:
            return torch.int64
        return operator.attrgetter(module)(torch).int64

    default_dtype = torch.get_default_dtype()
    check_value(torch.empty(shape), default_dtype, torch.strided, -1, None, False)
    check_value(torch.full(shape, -5), default_dtype, torch.strided, -1, None, False)
    for dtype in dtypes:
        for rg in {dtype.is_floating_point, False}:
            int64_dtype = get_int64_dtype(dtype)
            v = torch.empty(shape, dtype=dtype, device=device, layout=layout, requires_grad=rg)
            check_value(v, dtype, layout, device, None, rg)
            out = v.new()
            check_value(torch.empty(shape, out=out, device=device, layout=layout, requires_grad=rg),
                        dtype, layout, device, None, rg)
            check_value(v.new_empty(shape), dtype, layout, device, None, False)
            check_value(v.new_empty(shape, dtype=int64_dtype, device=device, requires_grad=False),
                        int64_dtype, layout, device, None, False)
            check_value(torch.empty_like(v), dtype, layout, device, None, False)
            check_value(torch.empty_like(v, dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
                        int64_dtype, layout, device, None, False)

            if dtype is not torch.float16 and layout != torch.sparse_coo:
                fv = 3
                v = torch.full(shape, fv, dtype=dtype, layout=layout, device=device, requires_grad=rg)
                check_value(v, dtype, layout, device, fv, rg)
                check_value(v.new_full(shape, fv + 1), dtype, layout, device, fv + 1, False)
                out = v.new()
                check_value(torch.full(shape, fv + 2, out=out, device=device, layout=layout, requires_grad=rg),
                            dtype, layout, device, fv + 2, rg)
                check_value(v.new_full(shape, fv + 3, dtype=int64_dtype, device=device, requires_grad=False),
                            int64_dtype, layout, device, fv + 3, False)
                check_value(torch.full_like(v, fv + 4), dtype, layout, device, fv + 4, False)
                check_value(torch.full_like(v, fv + 5,
                                            dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
                            int64_dtype, layout, device, fv + 5, False)

class TestEmptyMemoryFormat(TestCase):
    def test_empty_full(self, device):
        do_test_empty_full(self, torch.testing.get_all_math_dtypes('cpu'), torch.strided, torch.device('cpu'))
        if torch.cuda.device_count() > 0:
            do_test_empty_full(self, torch.testing.get_all_math_dtypes('cpu'), torch.strided, None)
            do_test_empty_full(self, torch.testing.get_all_math_dtypes('cpu'), torch.strided, torch.device('cuda:0'))

class TestContiguous(TestCase):
    def test_contiguous(self, device):
        x = torch.randn(1, 16, 5, 5, device="cpu").to(device)
        self.assertTrue(x.is_contiguous())
        stride = list(x.stride())
        stride[0] = 20
        # change the stride in dimension 0. the tensor is still contiguous because size[0] is 1
        x.set_(x.storage(), 0, x.size(), stride)
        self.assertTrue(x.is_contiguous())
    
    def test_contiguous_self(self, device):
        t = torch.ones(5, 5, device="cpu").to(device)
        s = t.contiguous()
        self.assertTrue(s is t)

    def test_contiguous_nonview(self, device):
        t = torch.ones(5, 5, device="cpu").to(device)
        nv = t.t().contiguous()
        self.assertTrue(not self.is_view_of(t, nv))

        nv[0, 0] = 0
        self.assertNotEqual(t[0, 0], nv[0, 0])

class TestCopy(TestCase):
    def test_copy_broadcast(self, device):
        torch.zeros(5, 6).copy_(torch.zeros(6))
        self.assertRaises(RuntimeError, lambda: torch.zeros(5, 6).copy_(torch.zeros(30)))

    def test_copy_many_to_one(self, device):
        # Testing in-place copy where it attempt to write from many memory
        # storage to a single storage would cause RuntimeError to be thrown
        self.assertRaises(RuntimeError, lambda: torch.zeros(1, 6).expand(5, 6).copy_(torch.zeros(5, 6)))

    def test_copy_all_dtypes_and_devices(self, device):
        from copy import copy
        for dt in torch.testing.get_all_dtypes():
            x = torch.tensor([1, 2, 3, 4], dtype=dt, device=device)
            x_clone = x.clone()
            y = copy(x)
            y.fill_(1)
            # copy is a shallow copy, only copies the tensor view,
            # not the data
            self.assertEqual(x, y)
    
    @dtypes(torch.double)
    def test_copy_mem_overlap(self, device, dtype):
        self.check_internal_mem_overlap(
            torch.Tensor.copy_, num_inputs=2, dtype=dtype, device=device)
        sz = 3
        doubles = torch.randn(2 * sz, dtype=dtype, device=device)
        self.unary_check_input_output_mem_overlap(
            doubles, sz, lambda input, out: out.copy_(input))
    
    #@deviceCountAtLeast(1)
    def test_copy_noncontig(self, devices):
        def do_test(d0, d1):
            x = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], device=d0)
            y = torch.tensor([0, 0, 0, 0, 0, 0], device=d1)
            self.assertNotEqual(x.dtype, y.dtype)

            y[::2].copy_(x[::2])
            self.assertEqual(y, [1, 0, 3, 0, 5, 0])

        do_test('cpu', devices[0])
        do_test(devices[0], 'cpu')

        if len(devices) > 1:
            do_test(devices[0], devices[1])
    
    def test_copy_broadcast(self, device):
        x = torch.randn(10, 5)
        y = torch.randn(5, device=device)
        x.copy_(y)
        self.assertEqual(x[3], y)

        x = torch.randn(10, 5, device=device)
        y = torch.randn(5)
        x.copy_(y)
        self.assertEqual(x[3], y)
    
    def test_copy_dtypes(self, device):
        all_dtypes = torch.testing.get_all_dtypes()
        for dtype in all_dtypes:
            copied_dtype = copy.deepcopy(dtype)
            self.assertIs(dtype, copied_dtype)

    def test_copy_transpose(self, device):
        x = torch.arange(100 * 100, dtype=torch.float).reshape(100, 100).t()
        y = torch.empty(100, 100, dtype=torch.float)
        y.copy_(x)
        self.assertEqual(y[:, 0], range(100))
        self.assertEqual(y[:, 40], range(4000, 4100))

        y = torch.empty(100, 100, dtype=torch.double)
        y.copy_(x)
        self.assertEqual(y[:, 0], range(100))
        self.assertEqual(y[:, 40], range(4000, 4100))

class TestTopK(TestCase):
    def test_topk(self, device):
        def topKViaSort(t, k, dim, dir):
            sorted, indices = t.sort(dim, dir)
            return sorted.narrow(dim, 0, k), indices.narrow(dim, 0, k)

        def compareTensors(t, res1, ind1, res2, ind2, dim):
            # Values should be exactly equivalent
            self.assertEqual(res1, res2, 0)

            # Indices might differ based on the implementation, since there is
            # no guarantee of the relative order of selection
            if not ind1.eq(ind2).all():
                # To verify that the indices represent equivalent elements,
                # gather from the input using the topk indices and compare against
                # the sort indices
                vals = t.gather(dim, ind2)
                self.assertEqual(res1, vals, 0)

        def compare(t, k, dim, dir):
            topKVal, topKInd = t.topk(k, dim, dir, True)
            sortKVal, sortKInd = topKViaSort(t, k, dim, dir)
            compareTensors(t, sortKVal, sortKInd, topKVal, topKInd, dim)

        t = torch.rand(random.randint(1, SIZE),
                       random.randint(1, SIZE),
                       random.randint(1, SIZE))

        for _kTries in range(3):
            for _dimTries in range(3):
                for transpose in (True, False):
                    for dir in (True, False):
                        testTensor = t
                        if transpose:
                            dim1 = random.randrange(t.ndimension())
                            dim2 = dim1
                            while dim1 == dim2:
                                dim2 = random.randrange(t.ndimension())

                            testTensor = t.transpose(dim1, dim2)

                        dim = random.randrange(testTensor.ndimension())
                        k = random.randint(1, testTensor.size(dim))
                        compare(testTensor, k, dim, dir)

    def test_topk_arguments(self, device):
        q = torch.randn(10, 2, 10).to(device)
        # Make sure True isn't mistakenly taken as the 2nd dimension (interpreted as 1)
        self.assertRaises(TypeError, lambda: q.topk(4, True))
    
    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    def test_topk_integral(self, device, dtype):
        #a = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, size=(10,),
                          #dtype=dtype, device=device)
        a = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, size=(10,),
                          dtype=dtype, device="cpu")
        sort_topk = a.sort()[0][-5:].flip(0)
        a = a.to("npu")
        topk = a.topk(5)
        topk = topk.to("cpu")
        self.assertEqual(sort_topk, topk[0])      # check values
        self.assertEqual(sort_topk, a[topk[1]])   # check indices

    #@dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_topk_nonfinite(self, device, dtype):
        x = torch.tensor([float('nan'), float('inf'), 1e4, 0, -1e4, -float('inf')], device=device, dtype=dtype)
        val, idx = x.topk(4)
        expect = torch.tensor([float('nan'), float('inf'), 1e4, 0], device=device, dtype=dtype)
        self.assertEqual(val.to("cpu"), expect.to("cpu"), allow_inf=True)
        self.assertEqual(idx.to("cpu"), [0, 1, 2, 3])

        val, idx = x.topk(4, largest=False)
        expect = torch.tensor([-float('inf'), -1e4, 0, 1e4], device=device, dtype=dtype)
        self.assertEqual(val.to("cpu"), expect.to("cpu"), allow_inf=True)
        self.assertEqual(idx.to("cpu"), [5, 4, 3, 2])

    #@dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_topk_nonfinite(self, device, dtype):
        x = torch.tensor([float('nan'), float('inf'), 1e4, 0, -1e4, -float('inf')], device=device, dtype=dtype)
        val, idx = x.topk(4)
        expect = torch.tensor([float('nan'), float('inf'), 1e4, 0], device=device, dtype=dtype)
        self.assertEqual(val.to("cpu"), expect.to("cpu"), allow_inf=True)
        self.assertEqual(idx.to("cpu"), [0, 1, 2, 3])

        val, idx = x.topk(4, largest=False)
        expect = torch.tensor([-float('inf'), -1e4, 0, 1e4], device=device, dtype=dtype)
        self.assertEqual(val.to("cpu"), expect.to("cpu"), allow_inf=True)
        self.assertEqual(idx.to("cpu"), [5, 4, 3, 2])

class TestStack(TestCase):
    def test_stack(self, device):
        for dtype in (torch.half, torch.double, torch.int):
            x = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype).to(device)
            y = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype).to(device)
            z = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype).to(device)
            for dim in range(4):
                res = torch.stack((x, y, z), dim)
                res_neg = torch.stack((x, y, z), dim - 4)
                expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
                self.assertEqual(res, res_neg)
                self.assertEqual(res.size(), expected_size)
                self.assertEqual(res.select(dim, 0), x, 0)
                self.assertEqual(res.select(dim, 1), y, 0)
                self.assertEqual(res.select(dim, 2), z, 0)

    def test_stack_out(self, device):
        for dtype in (torch.half, torch.double, torch.int):
            x = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype).to(device)
            y = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype).to(device)
            z = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype).to(device)
            for dim in range(4):
                expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
                res_out = x.new(expected_size)
                res_neg_out = x.new(expected_size)
                res_out_dp = res_out.data_ptr()
                res_out_neg_dp = res_neg_out.data_ptr()
                torch.stack((x, y, z), dim, out=res_out)
                torch.stack((x, y, z), dim - 4, out=res_neg_out)
                self.assertEqual(res_out, res_neg_out)
                self.assertEqual(res_out.size(), expected_size)
                self.assertEqual(res_out_dp, res_out.data_ptr())
                self.assertEqual(res_out_neg_dp, res_neg_out.data_ptr())
                self.assertEqual(res_out.select(dim, 0), x, 0)
                self.assertEqual(res_out.select(dim, 1), y, 0)
                self.assertEqual(res_out.select(dim, 2), z, 0)

class TestSetStorage(TestCase):
    def test_empty_strided(self, device):
        for shape in [(2, 3, 4), (0, 2, 0)]:
            # some of these cases are pretty strange, just verifying that if as_strided
            # allows them then empty_strided can as well.
            for strides in [(12, 4, 1), (2, 4, 6), (0, 0, 0)]:
                empty_strided = torch.empty_strided(shape, strides, device=device)
                # as_strided checks the storage size is big enough to support such a strided tensor;
                # instead of repeating this calculation, we just use empty_strided which does the same
                # calculation when setting the storage size.
                as_strided = torch.empty(empty_strided.storage().size(),
                                         device=device).as_strided(shape, strides)
                self.assertEqual(empty_strided.shape, as_strided.shape)
                self.assertEqual(empty_strided.stride(), as_strided.stride())

#class TestItem:
#class TestBatchNorm:
#class TestSetStorageOffset:
#class TestBroadcast:
#class TestArgmax:

instantiate_device_type_tests(TestAddcmul, globals(), except_for="cpu")
instantiate_device_type_tests(TestThreshold, globals(), except_for="cpu")
instantiate_device_type_tests(TestNllloss, globals(), except_for="cpu")
instantiate_device_type_tests(TestLogsoftmax, globals(), except_for="cpu")
instantiate_device_type_tests(TestOneslike, globals(), except_for="cpu")
instantiate_device_type_tests(TestOnes, globals(), except_for="cpu")
instantiate_device_type_tests(TestSoftmax, globals(), except_for="cpu")
instantiate_device_type_tests(TestAddmm, globals(), except_for="cpu")
instantiate_device_type_tests(Testmm, globals(), except_for="cpu")
instantiate_device_type_tests(TestTranspose, globals(), except_for="cpu")
instantiate_device_type_tests(TestT, globals(), except_for="cpu")
instantiate_device_type_tests(TestClone, globals(), except_for="cpu")
instantiate_device_type_tests(TestSqrt, globals(), except_for="cpu")
instantiate_device_type_tests(TestResize, globals(), except_for="cpu")

instantiate_device_type_tests(TestConv2d, globals(), except_for="cpu")
instantiate_device_type_tests(Testto, globals(), except_for="cpu")
instantiate_device_type_tests(TestMaxPool2d, globals(), except_for="cpu")
instantiate_device_type_tests(TestSet, globals(), except_for="cpu")
instantiate_device_type_tests(TestView, globals(), except_for="cpu")
instantiate_device_type_tests(TestEmptyMemoryFormat, globals(), except_for="cpu")
instantiate_device_type_tests(TestContiguous, globals(), except_for="cpu")
instantiate_device_type_tests(TestTopK, globals(), except_for="cpu")
instantiate_device_type_tests(TestCopy, globals(), except_for="cpu")
instantiate_device_type_tests(TestSetStorage, globals(), except_for="cpu")

if __name__ == "__main__":
    run_tests()