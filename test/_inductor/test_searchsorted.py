import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestSearchSorted(TestUtils):
    def eager_fn(self, offsets, values):
        boundaries = (offsets + 0)[1:]
        return torch.searchsorted(boundaries, values, right=True)

    def test_searchsorted(self):
        offsets = torch.tensor([0, 2, 5, 9, 14], dtype=torch.int64, device="npu")
        values = torch.tensor([0, 1, 2, 4, 5, 8, 13, 14], dtype=torch.int64, device="npu")
        expected = self.eager_fn(offsets, values)
        compile_fn = torch.compile(self.eager_fn, backend="inductor", dynamic=False)
        compile_result = compile_fn(offsets, values)

        self.assertEqual(expected, compile_result, atol=1e-1, rtol=1e-1)


instantiate_parametrized_tests(TestSearchSorted)

if __name__ == "__main__":
    run_tests()
