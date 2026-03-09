import torch
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu


def _make_fn_add_cumsum_sum(dim: int):
    def fn(x):
        a = x + 1
        b = torch.ops.aten.cumsum(a, dim=dim)
        c = torch.ops.aten.sum(b, dim=dim)
        return c

    return fn


def _make_fn_add_cumsum_sum_diff_dim(dim: int):
    def fn(x):
        a = x + 1
        b = torch.ops.aten.cumsum(a, dim=dim)
        c = torch.ops.aten.sum(b, dim=-1)
        return c

    return fn


def _make_fn_add_cumsum_add(dim: int):
    def fn(x):
        a = x + 1
        b = torch.ops.aten.cumsum(a, dim=dim)
        c = b + 1
        return c

    return fn


def _build_test_cases():
    # fn spec: (name, fn_factory, input_dtype, rtol, atol)
    fn_cases = [
        ("add_cumsum_sum", _make_fn_add_cumsum_sum, torch.int32, 0, 0),
        ("add_cumsum_add", _make_fn_add_cumsum_add, torch.int32, 0, 0),
        ("add_cumsum_sum_diff_dim", _make_fn_add_cumsum_sum_diff_dim, torch.int32, 0, 0),
    ]

    # shape/dim spec: ((shape), dim)
    shape_dim_cases = [
        ((3, 4, 5), 0),
        ((3, 4, 5), 1),
        ((3, 4, 5), -1),
        ((8192, 4, 5), 0),
        ((3, 8192, 5), 1),
        ((3, 4, 8192), -1),
        ((3, 4), 0),
        ((3, 4), -1),
    ]

    test_cases = []
    for shape, dim in shape_dim_cases:
        for fn_name, fn_factory, input_dtype, rtol, atol in fn_cases:
            test_cases.append((shape, dim, fn_name, fn_factory, input_dtype, rtol, atol))
    return test_cases


TEST_CASES = _build_test_cases()


class TestScan(TestUtils):
    @parametrize(
        "shape, dim, fn_name, fn_factory, input_dtype, rtol, atol",
        TEST_CASES,
    )
    def test_scan_aten_op(self, shape, dim, fn_name, fn_factory, input_dtype, rtol, atol):

        x = torch.ones(shape, device=torch.device("npu"), dtype=input_dtype)

        fn = fn_factory(dim)
        compiled = torch.compile(fn, backend="inductor", dynamic=False)

        out_inductor = compiled(x).to(torch.int32)
        out_eager = fn(x).to(torch.int32)
        max_abs_diff = (out_inductor - out_eager).abs().max().item()
        print(f"=== fn={fn_name} case={(shape, dim)} dtype={input_dtype} ===")
        print(f"allclose: {torch.allclose(out_inductor, out_eager, rtol=rtol, atol=atol)}")
        self.assertEqual(out_eager, out_inductor, rtol=rtol, atol=atol)


instantiate_parametrized_tests(TestScan)

if __name__ == "__main__":
    run_tests()
