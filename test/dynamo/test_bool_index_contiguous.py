# Owner(s): ["module: dynamo"]
import torch
from torch._dynamo.test_case import TestCase

import torch_npu  # noqa: F401


class TestBoolIndexContiguous(TestCase):
    def test_bool_index_slice_contiguous_fullgraph(self):
        def f(x):
            return x[x[:, 0] > 0][:, :2].contiguous()

        x = torch.ones(2, 3, device="npu")
        expected = f(x)
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            compiled = torch.compile(f, fullgraph=True)
            actual = compiled(x)

        self.assertEqual(actual, expected)
        self.assertEqual(tuple(actual.shape), (2, 2))
        self.assertTrue(actual.is_contiguous())


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
