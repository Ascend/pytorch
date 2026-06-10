# Owner(s): ["module: tests"]
import torch
import torch_npu.testing
import torch.utils.data
from torch.testing._internal.common_utils import run_tests, TestCase, IS_FBCODE, IS_REMOTE_GPU, skipIfRocm
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyPRIVATEUSE1, largeTensorTest
import unittest

class TestTensorShape(TestCase):
    def test_tensor_shape_empty(self, device):
        x = torch.randn((0, 1, 3, 0), device=device)
        # flatten
        self.assertEqual((0,), torch.flatten(x, 0, 3).shape)
        self.assertEqual((0, 0), torch.flatten(x, 0, 2).shape)
        self.assertEqual((0, 3, 0), torch.flatten(x, 1, 2).shape)

        # squeeze, unsqueeze
        self.assertEqual((0, 1, 1, 3, 0), torch.unsqueeze(x, 1).shape)
        self.assertEqual((0, 3, 0), torch.squeeze(x, 1).shape)
        self.assertEqual((0, 3, 0), torch.squeeze(x).shape)

        # transpose, t
        self.assertEqual((0, 0, 3, 1), torch.transpose(x, 1, 3).shape)
        y = torch.randn((5, 0), device=device)
        self.assertEqual((0, 5), y.t().shape)

        # select
        self.assertEqual((0, 1, 0), torch.select(x, 2, 2).shape)

        # repeat, permute
        self.assertEqual((9, 0, 5, 6, 0), x.repeat(9, 7, 5, 2, 3).shape)
        self.assertEqual((3, 0, 0, 1), x.permute(2, 3, 0, 1).shape)

        # diagonal, diagflat
        self.assertEqual((0,), torch.diagonal(torch.randn((5, 0), device=device)).shape)
        self.assertEqual((0,), torch.diagonal(torch.randn((0, 5), device=device)).shape)
        # off the end offsets are valid
        self.assertEqual((0,), torch.diagonal(torch.randn((5, 0), device=device), offset=1).shape)
        self.assertEqual((0,), torch.diagonal(torch.randn((0, 5), device=device), offset=1).shape)
        # check non-zero sized offsets off the end
        self.assertEqual((5, 6, 0), torch.diagonal(torch.randn((3, 4, 5, 6), device=device), offset=45252).shape)
        self.assertEqual((5, 6, 0), torch.diagonal(torch.randn((3, 4, 5, 6), device=device), offset=-45252).shape)

        self.assertEqual((0, 0), torch.diagflat(torch.tensor([], device=device)).shape)
        self.assertEqual(torch.zeros(1, 1), torch.diagflat(torch.tensor([], device=device), offset=1))
        self.assertEqual((0, 0), torch.diagflat(torch.tensor([[]], device=device)).shape)
        self.assertEqual(torch.zeros(1, 1), torch.diagflat(torch.tensor([[]], device=device), offset=1))

        # stack, split, chunk
        self.assertEqual((4, 0, 1, 3, 0), torch.stack((x, x, x, x)).shape)
        self.assertEqual([(0, 1, 3, 0)],
                         [z.shape for z in torch.chunk(x, 1, dim=0)])

        self.assertEqual([(0, 1, 3, 0), ] * 3, [z.shape for z in torch.chunk(x, 3, dim=0)])
        self.assertEqual([(0, 1, 1, 0), ] * 3, [z.shape for z in torch.chunk(x, 3, dim=2)])

        # NOTE: split_with_sizes behaves differently than NumPy in that it
        # takes sizes rather than offsets
        self.assertEqual([(0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 2, 0)],
                         [z.shape for z in torch.split(x, (0, 1, 2), dim=2)])

        self.assertRaises(RuntimeError, lambda: torch.split(x, 0, dim=1))
        # This is strange because the split size is larger than the dim size, but consistent with
        # how split handles that case generally (when no 0s are involved).
        self.assertEqual([(0, 1, 3, 0)], [z.shape for z in torch.split(x, 1, dim=0)])
        self.assertEqual([(0, 1, 3, 0)], [z.shape for z in torch.split(x, 0, dim=0)])

instantiate_device_type_tests(TestTensorShape, globals(), only_for='privateuse1')

if __name__ == "__main__":
    run_tests()