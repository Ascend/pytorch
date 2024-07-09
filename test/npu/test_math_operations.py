import copy
import numpy as np
import torch
from torch_npu.testing.testcase import TestCase, run_tests

import torch_npu

device = 'npu:0'
torch.npu.set_device(device)


class TestPointwiseOps(TestCase):

    def test_real(self):
        cpu_input = torch.randn(4, dtype=torch.float32)
        npu_input = cpu_input.npu()
        cpu_output = torch.real(cpu_input)
        npu_output = torch.real(npu_input)
        self.assertEqual(cpu_output, npu_output)

    def test_square(self):
        input1 = torch.randn(4)
        npu_input = input1.npu()
        cpu_output = torch.square(input1)
        npu_output = torch.square(npu_input)

        self.assertRtolEqual(npu_output.cpu().numpy(), cpu_output.numpy())


class TestReductionOps(TestCase):
    def test_dist(self):
        x = torch.zeros(3, device=device)
        y = torch.zeros(3, device=device)
        cpu_x = x.cpu()
        cpu_y = y.cpu()
        npu_output = torch.dist(x, y, 3)
        cpu_output = torch.dist(cpu_x, cpu_y, 3)
        self.assertRtolEqual(npu_output.cpu().numpy(), cpu_output.numpy())

    def test_unique(self):
        output = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.int32, device=device))
        output_expected = torch.tensor([1, 2, 3]).int()
        self.assertRtolEqual(output.cpu(), output_expected)


class TestComparisonOps(TestCase):
    def test_allclose(self):
        input1 = torch.tensor([10000., 1e-07])
        input2 = torch.tensor([10000.1, 1e-08])

        npu_input1 = input1.npu()
        npu_input2 = input2.npu()

        npu_output = torch.allclose(npu_input1, npu_input2)

        self.assertFalse(npu_output)


class TestSpectralOps(TestCase):
    def test_isinf(self):
        input1 = torch.randn(1, device=device)
        output = torch.isinf(input1)

        self.assertFalse(output)


class TestOtherOps(TestCase):
    def test_broadcast_tensors(self):
        x = torch.arange(3).view(1, 3).npu()
        y = torch.arange(2).view(2, 1).npu()
        a, b = torch.broadcast_tensors(x, y)
        expected_cpu_a = torch.tensor([[0, 1, 2], [0, 1, 2]])
        expected_cpu_b = torch.tensor([[0, 0, 0], [1, 1, 1]])
        self.assertEqual(a.cpu(), expected_cpu_a)
        self.assertEqual(b.cpu(), expected_cpu_b)

    def test_cartesian_prod(self):
        a = [1, 2, 3]
        b = [4, 5]
        import itertools
        output_expected = torch.tensor(list(itertools.product(a, b)))

        tensor_a = torch.tensor(a, device=device)
        tensor_b = torch.tensor(b, device=device)
        output = torch.cartesian_prod(tensor_a, tensor_b)
        self.assertRtolEqual(output_expected, output.cpu())

    def test_einsum(self):
        input1 = torch.randn(5)
        input2 = torch.randn(4)

        npu_input1 = copy.deepcopy(input1).npu()
        npu_input2 = copy.deepcopy(input2).npu()

        cpu_output = torch.einsum('i,j->ij', input1, input2)
        npu_output = torch.einsum('i,j->ij', npu_input1, npu_input2)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_rot90(self):
        x = torch.arange(4, device=device).view(2, 2)
        output = torch.rot90(x, 1, [0, 1]).int()
        output_expected = torch.tensor([[1, 3], [0, 2]]).int()
        self.assertRtolEqual(output.cpu(), output_expected)

    def test_flatten(self):
        x = torch.randn((0, 1, 3, 0), device=device)
        self.assertEqual((0,), torch.flatten(x, 0, 3).shape)
        self.assertEqual((0, 0), torch.flatten(x, 0, 2).shape)
        self.assertEqual((0, 3, 0), torch.flatten(x, 1, 2).shape)

    def test_meshgrid(self):
        a = torch.tensor(1, device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = torch.meshgrid([a, b, c])
        self.assertEqual(grid_a.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c.shape, torch.Size([1, 3, 2]))
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c)
        self.assertEqual(grid_a2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c2.shape, torch.Size([1, 3, 2]))
        expected_grid_a = torch.ones(1, 3, 2, dtype=torch.int64, device=device)
        expected_grid_b = torch.tensor([[[1, 1],
                                         [2, 2],
                                         [3, 3]]], device=device)
        expected_grid_c = torch.tensor([[[1, 2],
                                         [1, 2],
                                         [1, 2]]], device=device)
        self.assertTrue(grid_a.equal(expected_grid_a))
        self.assertTrue(grid_b.equal(expected_grid_b))
        self.assertTrue(grid_c.equal(expected_grid_c))
        self.assertTrue(grid_a2.equal(expected_grid_a))
        self.assertTrue(grid_b2.equal(expected_grid_b))
        self.assertTrue(grid_c2.equal(expected_grid_c))

    def test_tensordot(self):
        a = torch.arange(60., device=device).reshape(3, 4, 5)
        b = torch.arange(24., device=device).reshape(4, 3, 2)
        npu_output = torch.tensordot(a, b, dims=([1, 0], [0, 1]))

        a = a.cpu()
        b = b.cpu()
        cpu_output = torch.tensordot(a, b, dims=([1, 0], [0, 1]))
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())


class TestBLOps(TestCase):
    def test_chain_matmul(self):
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        c = torch.randn(5, 6)
        d = torch.randn(6, 7)
        cpu_output = torch.chain_matmul(a, b, c, d)
        npu_output = torch.chain_matmul(a.half().npu(), b.half().npu(), c.half().npu(), d.half().npu())
        self.assertEqual(cpu_output, npu_output, prec=1e-2)

    def test_trapz(self):
        def test_dx(sizes, dim, dx):
            t = torch.randn(sizes, device=device)
            actual = torch.trapz(t, dx=dx, dim=dim)
            expected = np.trapz(t.cpu().numpy(), dx=dx, axis=dim)
            self.assertEqual(expected.shape, actual.shape)
            self.assertTrue(np.allclose(expected, actual.cpu().numpy()))

        def test_x(sizes, dim, x):
            t = torch.randn(sizes, device=device)
            actual = torch.trapz(t, x=torch.tensor(x, device=device), dim=dim)
            expected = np.trapz(t.cpu().numpy(), x=x, axis=dim)
            self.assertEqual(expected.shape, actual.shape)
            self.assertTrue(np.allclose(expected, actual.cpu().numpy()))

        test_dx((2, 3, 4), 1, 1)
        test_dx((10, 2), 0, 0.1)
        test_dx((1, 10), 0, 2.3)
        test_dx((0, 2), 0, 1.0)
        test_dx((0, 2), 1, 1.0)
        test_x((2, 3, 4), 1, [1.0, 2.0, 3.0])
        test_x((10, 2), 0, [2.0, 3.0, 4.0, 7.0, 11.0, 14.0, 22.0, 26.0, 26.1, 30.3])
        test_x((1, 10), 0, [1.0])
        test_x((0, 2), 0, [])
        test_x((0, 2), 1, [1.0, 2.0])
        with self.assertRaisesRegex(
                IndexError,
                'Dimension out of range'):
            test_x((2, 3), 2, [])
            test_dx((2, 3), 2, 1.0)
        with self.assertRaisesRegex(
                RuntimeError,
                'There must be one `x` value for each sample point'):
            test_x((2, 3), 1, [1.0, 2.0])
            test_x((2, 3), 1, [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    run_tests()
