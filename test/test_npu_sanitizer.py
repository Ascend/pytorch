# Owner(s): ["module: npu"]

import sys

import torch
import torch.cuda._sanitizer as csan
from torch.testing._internal.common_utils import NoTest, run_tests, TEST_PRIVATEUSE1, TestCase
from torch_npu.npu._stream_check import apply_sanitizer_patch


if not TEST_PRIVATEUSE1:
    print("NPU not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


class TestArgumentHandler(TestCase):
    def test_add(self):
        add_func = torch.ops.aten.add.Tensor
        a = torch.ones(5, 3, device="npu")
        b = torch.randn(5, 3, device="npu")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(add_func._schema, (a, b), {}, is_factory=False)
        c = torch.add(a, b)
        argument_handler.parse_outputs(add_func._schema, c, is_factory=False)

        self.assertEqual({a.data_ptr(), b.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual({c.data_ptr()}, argument_handler.dataptrs_written)

    def test_cat(self):
        cat_func = torch.ops.aten.cat.default
        a = torch.ones(2, 4, 5, device="npu")
        b = torch.zeros(2, 1, 5, device="npu")
        c = torch.rand(2, 7, 5, device="npu")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            cat_func._schema, ([a, b, c], 1), {}, is_factory=False
        )
        d = torch.cat((a, b, c), dim=1)
        argument_handler.parse_outputs(cat_func._schema, d, is_factory=False)

        self.assertEqual(
            {a.data_ptr(), b.data_ptr(), c.data_ptr()}, argument_handler.dataptrs_read
        )
        self.assertEqual({d.data_ptr()}, argument_handler.dataptrs_written)

    def test_split(self):
        split_func = torch.ops.aten.split.Tensor
        a = torch.arange(10, device="npu").reshape(5, 2)

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(split_func._schema, (a, 2), {}, is_factory=False)
        out = torch.split(a, 2)
        argument_handler.parse_outputs(split_func._schema, out, is_factory=False)

        outputs = {out[0].data_ptr(), out[1].data_ptr(), out[2].data_ptr()}
        # Split is a view op, no data is read or written!
        self.assertEqual(len(argument_handler.dataptrs_read), 0)
        self.assertEqual(len(argument_handler.dataptrs_written), 0)

    def test_inplace(self):
        add_inplace_func = torch.ops.aten.add_.Tensor
        a = torch.rand(4, 2, device="npu")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            add_inplace_func._schema, (a, 5), {}, is_factory=False
        )
        a.add_(5)
        argument_handler.parse_outputs(add_inplace_func._schema, a, is_factory=False)

        self.assertEqual(set(), argument_handler.dataptrs_read)
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_written)

    def test_out(self):
        mul_out_func = torch.ops.aten.mul.out
        a = torch.arange(8, device="npu")
        b = torch.empty(8, device="npu")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            mul_out_func._schema, (a, 3), {"out": b}, is_factory=False
        )
        torch.mul(a, 3, out=b)
        argument_handler.parse_outputs(mul_out_func._schema, b, is_factory=False)

        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual({b.data_ptr()}, argument_handler.dataptrs_written)

    def test_nonzero(self):
        nonzero_func = torch.ops.aten.nonzero.default
        a = torch.ones(5, 3, 2, device="npu")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            nonzero_func._schema, (a,), {"as_tuple": True}, is_factory=False
        )
        out = torch.nonzero(a, as_tuple=True)
        argument_handler.parse_outputs(nonzero_func._schema, out, is_factory=False)

        outputs = {out[0].data_ptr(), out[1].data_ptr(), out[2].data_ptr()}
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual(outputs, argument_handler.dataptrs_written)

    def test_tensor_names(self):
        addr_func = torch.ops.aten.addr.default
        vec = torch.arange(1, 4, device="npu")
        M = torch.zeros(3, 3, device="npu")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            addr_func._schema, (M, vec, vec), {}, is_factory=False
        )
        out = torch.addr(M, vec, vec)
        argument_handler.parse_outputs(addr_func._schema, out, is_factory=False)

        self.assertEqual(
            argument_handler.tensor_aliases,
            {
                M.data_ptr(): ["self"],
                vec.data_ptr(): ["vec1", "vec2"],
                out.data_ptr(): [],
            },
        )
        self.assertEqual({out.data_ptr()}, argument_handler.outputs)


if __name__ == "__main__":
    apply_sanitizer_patch()
    run_tests()