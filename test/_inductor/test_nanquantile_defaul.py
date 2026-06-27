import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
from torch._inductor.utils import run_and_get_code


class TestNanquantileDefault(TestUtils):

    @parametrize("dynamic", [False])
    @parametrize('shape', [(3, 4)])
    @parametrize('dtype', ['float32'])
    def test_nanquantile(self, shape, dtype, dynamic):
        a = torch.randn(shape, dtype=torch.float32, device="npu")
        a[0, 0] = float('nan')
        q = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32, device="npu")

        def fn(a, q):
            return torch.ops.aten.nanquantile.default(a, q, dim=0, keepdim=False)

        r1 = fn(a, q)
        func = torch.compile(fn, backend="inductor", dynamic=dynamic)
        r, codes = run_and_get_code(func, a, q)
        self.assertEqual(r, r1, atol=1e-3, rtol=1e-3)
        self.assertTrue('nanquantile' in codes[0])


instantiate_parametrized_tests(TestNanquantileDefault)


if __name__ == "__main__":
    run_tests()
