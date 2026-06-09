# Owner(s): ["module: tests"]
import torch
import torch_npu.testing
import torch.utils.data
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests

class TestPairwiseDistance(TestCase):
    def test_pairwise_distance_empty(self, device):
        shape = (2, 0)
        x = torch.randn(shape, device=device)
        y = torch.randn(shape, device=device)

        self.assertEqual(torch.zeros(2, device=device), torch.pairwise_distance(x, y))
        self.assertEqual(torch.zeros((2, 1), device=device), torch.pairwise_distance(x, y, keepdim=True))

        shape = (0, 2)
        x = torch.randn(shape, device=device)
        y = torch.randn(shape, device=device)
        self.assertEqual(torch.zeros(0, device=device), torch.pairwise_distance(x, y))
        self.assertEqual(torch.zeros((0, 1), device=device), torch.pairwise_distance(x, y, keepdim=True))

instantiate_device_type_tests(TestPairwiseDistance, globals(), only_for='privateuse1')

if __name__ == "__main__":
    run_tests()