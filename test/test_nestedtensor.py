# Owner(s): ["module: nested tensor"]

import torch
import numpy as np
from torch.testing._internal.common_utils import parametrize, instantiate_parametrized_tests
import torch_npu  # noqa: F401  (registers the npu device and its kernels)
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.utils import _jagged_ops


class TestNestedTensor(TestCase):
    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [16, 32])
    def test_2d_nested_tensor(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(1, max_seq_len)
            row = list(np.random.randint(low=0, high=vocab_size, size=(length,)))
            data.append(row)
            nested_tensor_ref_list.append(torch.tensor(row).npu())
        nested_tensor = torch.nested.nested_tensor(data)
        nested_tensor_list = nested_tensor.unbind()
        for i in range(batch_size):
            self.assertEqual(nested_tensor_list[i], nested_tensor_ref_list[i].type(torch.int64))

    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [16, 32])
    def test_3d_nested_tensor(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(1, max_seq_len)
            row = list(np.random.randint(low=0, high=vocab_size, size=(length,)))
            row = [list(item * np.arange(max_seq_len)) for item in row]
            data.append(row)
            nested_tensor_ref_list.append(torch.tensor(row).npu())
        nested_tensor = torch.nested.nested_tensor(data)
        nested_tensor_list = nested_tensor.unbind()
        for i in range(batch_size):
            self.assertEqual(nested_tensor_list[i], nested_tensor_ref_list[i].type(torch.int64))

    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [16, 32])
    def test_3d_nested_tensor_float(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(1, max_seq_len)
            row = list(np.random.randint(low=0, high=vocab_size, size=(length,)))
            row = [list(item * np.arange(max_seq_len)) for item in row]
            data.append(row)
            nested_tensor_ref_list.append(torch.tensor(row).npu())
        nested_tensor = torch.nested.nested_tensor(data)
        nested_tensor_list = nested_tensor.unbind()
        for i in range(batch_size):
            self.assertEqual(nested_tensor_list[i], nested_tensor_ref_list[i].type(torch.float32))

    def _test_unbind_case(self, a, b):
        nt = torch.nested.nested_tensor([a.npu(), b.npu()], dtype=a.dtype)
        nt_list = nt.unbind()
        self.assertEqual(len(nt_list), 2)
        self.assertEqual(nt_list[0], a)
        self.assertEqual(nt_list[1], b)

    def _test_asnested_case(self, a, b):
        nt = torch.nested.nested_tensor([a.npu(), b.npu()], dtype=a.dtype)
        nt_as = torch.nested.as_nested_tensor([a.npu(), b.npu()], dtype=a.dtype)
        nt_list = nt.unbind()
        nt_as_list = nt_as.unbind()
        self.assertEqual(len(nt_as_list), len(nt_list))
        self.assertEqual(nt_as_list[0], nt_list[0])
        self.assertEqual(nt_as_list[1], nt_list[1])

    def test_unbind_and_asnested_int64(self):
        a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        b = torch.tensor([[7, 8], [10, 11]])
        self._test_unbind_case(a, b)
        self._test_asnested_case(a, b)

    def test_unbind_and_asnested_float32(self):
        a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        b = torch.tensor([[7, 8], [10, 11]], dtype=torch.float32)
        self._test_unbind_case(a, b)
        self._test_asnested_case(a, b)

    def test_unbind_and_asnested_empty(self):
        a = torch.tensor([[], []])
        b = torch.tensor([[], [], []])
        self._test_unbind_case(a, b)
        self._test_asnested_case(a, b)

    def test_default_options_nested_tensor(self):
        default_nested_tensor = torch.nested.nested_tensor([], device="npu:0")
        default_tensor = torch.tensor([]).npu()
        self.assertEqual(default_nested_tensor.dtype, default_tensor.dtype)
        self.assertEqual(default_nested_tensor.device, default_tensor.device)
        self.assertEqual(default_nested_tensor.layout, default_tensor.layout)
        self.assertEqual(default_nested_tensor.dim(), default_tensor.dim())
        self.assertEqual(default_nested_tensor.requires_grad, default_tensor.requires_grad)

    def test_nested_tensor_errsize(self):
        nt = torch.nested.nested_tensor([torch.tensor([[1, 2, 3], [4, 5, 6]]).npu(), torch.tensor([[7, 8], [10, 11], [12, 13]]).npu()])
        self.assertEqual(nt.size(0), 2)
        self.assertRaisesRegex(RuntimeError,
        "Given dimension 1 is irregular and does not have a size",
        lambda: nt.size(1),
        )

        nt = torch.nested.nested_tensor([2])
        self.assertEqual(nt.size(0), 1)


JAGGED_LENGTHS = ([5, 9, 3], [7], [0], [0, 0], [4, 4], [0, 3, 0, 6], [2, 5])
JAGGED_DTYPES = (torch.float32, torch.bfloat16, torch.int64, torch.bool)
PADDING = {torch.float32: 0.0, torch.bfloat16: 0.0, torch.int64: -1.0, torch.bool: False}


def _make_jagged_parts(lengths, dtype, shape_tail=(4,)):
    return [_random_tensor((length, *shape_tail), dtype) for length in lengths]


def _random_tensor(shape, dtype):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.int64).to(dtype)
    return torch.randint(-8, 8, shape, dtype=dtype)


class TestJaggedNestedTensorOps(TestCase):
    """jagged NestedTensor packing ops must be executed on NPU, not via CPU fallback."""

    @staticmethod
    def _assert_same(got, expected, dtype):
        if dtype.is_floating_point:
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
        else:
            assert torch.equal(got, expected)

    def test_conversion_kernels_registered_for_npu(self):
        for op_name in ("aten::_jagged_to_padded_dense_forward", "aten::_padded_dense_to_jagged_forward"):
            self.assertTrue(
                torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "PrivateUse1"),
                f"{op_name} has no NPU kernel, it would fall back to the CPU",
            )

    def test_to_padded_tensor_jagged(self):
        for dtype in JAGGED_DTYPES:
            for lengths in JAGGED_LENGTHS:
                with self.subTest(dtype=dtype, lengths=lengths):
                    parts = _make_jagged_parts(lengths, dtype)
                    padding = PADDING[dtype]
                    expected = torch.nested.to_padded_tensor(
                        torch.nested.as_nested_tensor(parts, layout=torch.jagged), padding=padding)

                    nested = torch.nested.as_nested_tensor(parts, layout=torch.jagged).to("npu:0")
                    got = torch.nested.to_padded_tensor(nested, padding=padding)
                    self.assertEqual(got.device.type, "npu")
                    self._assert_same(got.cpu(), expected, dtype)
                    for index, length in enumerate(lengths):
                        pad_region = got[index, length:]
                        self._assert_same(pad_region.cpu(), torch.full_like(pad_region.cpu(), padding), dtype)

    def test_jagged_to_padded_dense_forward_op(self):
        for dtype in JAGGED_DTYPES:
            for lengths in JAGGED_LENGTHS:
                for shape_tail in ((), (3,)):
                    with self.subTest(dtype=dtype, lengths=lengths, shape_tail=shape_tail):
                        parts = _make_jagged_parts(lengths, dtype, shape_tail)
                        offsets = torch.tensor([0] + list(torch.cumsum(
                            torch.tensor(lengths), 0).tolist()), dtype=torch.int64)
                        max_len = max(lengths)
                        values = torch.cat(parts)
                        padding = PADDING[dtype]
                        expected = torch.ops.aten._jagged_to_padded_dense_forward(
                            values, [offsets], [max_len], padding)

                        got = torch.ops.aten._jagged_to_padded_dense_forward(
                            values.npu(), [offsets.npu()], [max_len], padding)
                        self.assertEqual(got.device.type, "npu")
                        self.assertEqual(got.shape, expected.shape)
                        self._assert_same(got.cpu(), expected, dtype)

    def test_padded_dense_to_jagged_forward_op(self):
        for dtype in JAGGED_DTYPES:
            for lengths in JAGGED_LENGTHS:
                with self.subTest(dtype=dtype, lengths=lengths):
                    padded = _random_tensor((len(lengths), max(lengths), 3), dtype)
                    offsets = torch.tensor([0] + list(torch.cumsum(
                        torch.tensor(lengths), 0).tolist()), dtype=torch.int64)
                    total_l = int(offsets[-1])
                    expected = torch.ops.aten._padded_dense_to_jagged_forward(
                        padded, [offsets], total_l)

                    got = torch.ops.aten._padded_dense_to_jagged_forward(
                        padded.npu(), [offsets.npu()], total_l)
                    self.assertEqual(got.device.type, "npu")
                    self.assertEqual(got.shape, expected.shape)
                    self._assert_same(got.cpu(), expected, dtype)

    def test_jagged_to_padded_dense_truncates_long_sequences(self):
        # Sequences longer than max_lengths are truncated instead of raising,
        # exactly like the CPU kernel; the padding positions stay untouched.
        lengths = [5, 9, 3]
        values = torch.cat(_make_jagged_parts(lengths, torch.float32, (2,)))
        offsets = torch.tensor([0] + list(torch.cumsum(
            torch.tensor(lengths), 0).tolist()), dtype=torch.int64)
        for max_len in (1, 4):
            with self.subTest(max_len=max_len):
                expected = torch.ops.aten._jagged_to_padded_dense_forward(
                    values, [offsets], [max_len], -1.0)
                got = torch.ops.aten._jagged_to_padded_dense_forward(
                    values.to("npu:0"), [offsets.to("npu:0")], [max_len], -1.0)
                self.assertEqual(got.device.type, "npu")
                self.assertEqual(got.shape, expected.shape)
                torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)

    def test_jagged_softmax(self):
        # The ragged softmax is built on both packing ops, so it must stay on
        # NPU as well.
        lengths = [5, 9, 3]
        nested = torch.nested.as_nested_tensor(
            _make_jagged_parts(lengths, torch.float32, (3,)), layout=torch.jagged)
        expected = torch.softmax(nested, dim=-1)
        got = torch.softmax(nested.to("npu:0"), dim=-1)
        self.assertEqual(got.values().device.type, "npu")
        # softmax does real arithmetic on the device, so this comparison keeps
        # the default floating point tolerance.
        torch.testing.assert_close(got.values().cpu(), expected.values())

    def test_padded_to_nested_to_padded_roundtrip(self):
        lengths = [5, 9, 3]
        padded = torch.randn(len(lengths), max(lengths), 3)
        offsets = torch.tensor([0] + list(torch.cumsum(
            torch.tensor(lengths), 0).tolist()), dtype=torch.int64)
        mask = torch.zeros_like(padded)
        for index, length in enumerate(lengths):
            mask[index, :length] = 1.0

        values = torch.ops.aten._padded_dense_to_jagged_forward(padded.npu(), [offsets.npu()], int(offsets[-1]))
        nested = torch.nested.nested_tensor_from_jagged(values, offsets.npu(), max_seqlen=max(lengths))
        got = torch.nested.to_padded_tensor(nested, padding=0.0)

        self.assertEqual(got.device.type, "npu")
        torch.testing.assert_close(got.cpu(), padded * mask, rtol=0, atol=0)

    def test_to_padded_tensor_backward(self):
        lengths = [5, 9, 3]
        parts = _make_jagged_parts(lengths, torch.float32, (3,))
        nested = torch.nested.as_nested_tensor(parts, layout=torch.jagged)
        values = nested.values().clone().npu().requires_grad_(True)
        offsets = nested.offsets().npu()

        nt = torch.nested.nested_tensor_from_jagged(values, offsets, max_seqlen=nested._max_seqlen)
        padded = torch.nested.to_padded_tensor(nt, padding=0.0)
        grad = torch.randn_like(padded)
        padded.backward(grad)

        self.assertIsNotNone(values.grad)
        self.assertEqual(values.grad.device.type, "npu")
        expected = torch.cat([grad.cpu()[index, :length] for index, length in enumerate(lengths)])
        torch.testing.assert_close(values.grad.cpu(), expected, rtol=0, atol=0)

    def test_conversions_run_the_npu_kernels(self):
        lengths = [5, 9, 3]
        parts = _make_jagged_parts(lengths, torch.float32, (3,))
        offsets = torch.tensor([0] + list(torch.cumsum(
            torch.tensor(lengths), 0).tolist()), dtype=torch.int64)
        nested = torch.nested.as_nested_tensor(parts, layout=torch.jagged)
        expected_padded = torch.nested.to_padded_tensor(nested, padding=0.0)
        nested = nested.to("npu:0")
        # The registered kernels resolve the implementation at call time, so the
        # recorded calls prove the NPU path was taken instead of the torch_npu
        # CPU fallback (whose one-shot TORCH_WARN is not a Python warning).
        calls = []
        originals = {}

        def counting(name, impl):
            def wrapper(*args, **kwargs):
                calls.append((name, args[0].device.type))
                return impl(*args, **kwargs)
            return wrapper

        for name in ("jagged_to_padded_dense", "padded_dense_to_jagged"):
            originals[name] = getattr(_jagged_ops, name)
            setattr(_jagged_ops, name, counting(name, originals[name]))
        try:
            padded = torch.nested.to_padded_tensor(nested, padding=0.0)
            values = torch.ops.aten._padded_dense_to_jagged_forward(
                padded, [offsets.npu()], int(offsets[-1]))
            torch.ops.aten._jagged_to_padded_dense_forward(
                values, [offsets.npu()], [max(lengths)], 0.0)
            torch.npu.synchronize()
        finally:
            for name, impl in originals.items():
                setattr(_jagged_ops, name, impl)

        self.assertEqual(
            {name for name, _ in calls},
            {"jagged_to_padded_dense", "padded_dense_to_jagged"},
            "both jagged kernels must run when converting to / from padded dense",
        )
        self.assertEqual({device for _, device in calls}, {"npu"})
        torch.testing.assert_close(padded.cpu(), expected_padded, rtol=0, atol=0)


class TestJaggedOpsComposition(TestCase):
    """The composed implementations must match the upstream CPU reference exactly."""

    def test_jagged_to_padded_dense_matches_cpu_reference(self):
        for dtype in JAGGED_DTYPES:
            for lengths in JAGGED_LENGTHS:
                with self.subTest(dtype=dtype, lengths=lengths):
                    values = torch.cat(_make_jagged_parts(lengths, dtype, (3,)))
                    offsets = torch.tensor([0] + list(torch.cumsum(
                        torch.tensor(lengths), 0).tolist()), dtype=torch.int64)
                    padding = PADDING[dtype]
                    expected = torch.ops.aten._jagged_to_padded_dense_forward(
                        values, [offsets], [max(lengths)], padding)
                    got = _jagged_ops.jagged_to_padded_dense(
                        values, [offsets], [max(lengths)], padding)
                    self.assertEqual(got.shape, expected.shape)
                    if dtype.is_floating_point:
                        torch.testing.assert_close(got, expected, rtol=0, atol=0)
                    else:
                        assert torch.equal(got, expected)

    def test_padded_dense_to_jagged_matches_cpu_reference(self):
        for dtype in JAGGED_DTYPES:
            for lengths in JAGGED_LENGTHS:
                with self.subTest(dtype=dtype, lengths=lengths):
                    padded = torch.randn(len(lengths), max(lengths), 3)
                    padded = padded.to(dtype) if dtype.is_floating_point else padded.gt(0).to(dtype)
                    offsets = torch.tensor([0] + list(torch.cumsum(
                        torch.tensor(lengths), 0).tolist()), dtype=torch.int64)
                    total_l = int(offsets[-1])
                    expected = torch.ops.aten._padded_dense_to_jagged_forward(padded, [offsets], total_l)
                    got = _jagged_ops.padded_dense_to_jagged(
                        padded, [offsets], total_l)
                    self.assertEqual(got.shape, expected.shape)
                    if dtype.is_floating_point:
                        torch.testing.assert_close(got, expected, rtol=0, atol=0)
                    else:
                        assert torch.equal(got, expected)

    def test_padding_sentinels_match_cpu_reference(self):
        offsets = torch.tensor([0, 2, 6], dtype=torch.int64)
        # inf / -inf are the "pad with dtype max / min" sentinels: they must be
        # clamped instead of overflowing the float -> int conversion.  For bool
        # that means +inf -> True and -inf -> False (not Python's bool(-inf)).
        for dtype in (torch.int64, torch.float32, torch.bfloat16, torch.bool):
            for padding in (float("inf"), float("-inf"), -1.0):
                with self.subTest(dtype=dtype, padding=padding):
                    values = _random_tensor((6, 3), dtype)
                    expected = torch.ops.aten._jagged_to_padded_dense_forward(
                        values, [offsets], [4], padding)
                    got = _jagged_ops.jagged_to_padded_dense(
                        values, [offsets], [4], padding)
                    if dtype.is_floating_point:
                        torch.testing.assert_close(got, expected, rtol=0, atol=0)
                    else:
                        assert torch.equal(got, expected)

    def test_sequences_longer_than_max_lengths(self):
        offsets = torch.tensor([0, 2, 6], dtype=torch.int64)
        values = torch.arange(6.0).reshape(6, 1)
        for max_len in (0, 1, 2, 4):
            with self.subTest(max_len=max_len):
                expected = torch.ops.aten._jagged_to_padded_dense_forward(
                    values, [offsets], [max_len], -1.0)
                got = _jagged_ops.jagged_to_padded_dense(values, [offsets], [max_len], -1.0)
                self.assertEqual(got.shape, expected.shape)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

        # The reverse direction rejects a ragged length the padded input cannot
        # hold and reports a total_L mismatch, in the same order as the CPU
        # kernel.
        padded = torch.randn(2, 2, 1)
        with self.assertRaisesRegex(RuntimeError, "found batch item of length 4"):
            _jagged_ops.padded_dense_to_jagged(padded, [offsets])
        with self.assertRaisesRegex(RuntimeError, "final offset should match total_L"):
            _jagged_ops.padded_dense_to_jagged(padded, [offsets], 5)
        with self.assertRaisesRegex(RuntimeError, "found batch item of length 4"):
            _jagged_ops.padded_dense_to_jagged(padded, [offsets], 6)
        self.assertEqual(
            _jagged_ops.padded_dense_to_jagged(torch.randn(2, 4, 1), [offsets], 6).shape, (6, 1))

    def test_empty_batch_matches_cpu_reference(self):
        offsets = torch.tensor([0], dtype=torch.int64)
        values = torch.randn(0, 3)
        expected = torch.ops.aten._jagged_to_padded_dense_forward(values, [offsets], [0], 0.0)
        got = _jagged_ops.jagged_to_padded_dense(values, [offsets], [0], 0.0)
        self.assertEqual(got.shape, expected.shape)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

        padded = torch.randn(0, 0, 3)
        expected = torch.ops.aten._padded_dense_to_jagged_forward(padded, [offsets], 0)
        got = _jagged_ops.padded_dense_to_jagged(padded, [offsets], 0)
        self.assertEqual(got.shape, expected.shape)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_unsupported_inputs_are_rejected(self):
        # Multi-dim jagged tensors are rejected by the CPU kernel as well;
        # NotImplementedError is a RuntimeError subclass, like upstream's raise.
        offsets = [torch.tensor([0, 2, 6]), torch.tensor([0, 1])]
        with self.assertRaises(RuntimeError):
            _jagged_ops.jagged_to_padded_dense(torch.randn(6), offsets, [3, 2], 0.0)
        with self.assertRaises(RuntimeError):
            _jagged_ops.padded_dense_to_jagged(torch.randn(2, 3), offsets, 6)
        with self.assertRaises(RuntimeError):
            _jagged_ops.padded_dense_to_jagged(
                torch.randn(2, 4, 3), [torch.tensor([0, 2, 6])], 5)

    def test_kernel_entry_points_delegate(self):
        # The registered PrivateUse1 entry points must reach the composed
        # implementations; the lookup happens at call time so that the NPU
        # tests can observe the dispatch.
        offsets = torch.tensor([0, 2, 6], dtype=torch.int64)
        values = torch.randn(6, 3)
        expected = torch.ops.aten._jagged_to_padded_dense_forward(values, [offsets], [4], 0.0)
        padded = _jagged_ops._jagged_to_padded_dense_kernel(values, [offsets], [4], 0.0)
        torch.testing.assert_close(padded, expected, rtol=0, atol=0)
        expected = torch.ops.aten._padded_dense_to_jagged_forward(padded, [offsets], 6)
        torch.testing.assert_close(
            _jagged_ops._padded_dense_to_jagged_kernel(padded, [offsets], 6), expected,
            rtol=0, atol=0)

        calls = []
        original = _jagged_ops.jagged_to_padded_dense

        def counting(*args, **kwargs):
            calls.append(args[0].device.type)
            return original(*args, **kwargs)

        _jagged_ops.jagged_to_padded_dense = counting
        try:
            _jagged_ops._jagged_to_padded_dense_kernel(values, [offsets], [4], 0.0)
        finally:
            _jagged_ops.jagged_to_padded_dense = original
        self.assertEqual(calls, ["cpu"])


if __name__ == '__main__':
    instantiate_parametrized_tests(TestNestedTensor)
    run_tests()
