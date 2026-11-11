# Owner(s): ["module: c10d"]
"""Core HCCL behavior. Correct results do not prove a CCU_MS algorithm hit."""
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol
from torch.testing._internal.common_utils import run_tests
from torch.utils.dlpack import from_dlpack, to_dlpack

from functional_all_reduce_test_utils import NpuCollectiveTestCase, assert_close, premul_reference
from torch_npu.distributed import functional_all_reduce


class TestFunctionalAllReduce(NpuCollectiveTestCase):
    def test_forward_and_independent_output(self):
        cases = [(torch.float32, op) for op in ("sum", "avg", "min", "max")]
        # Representative conversion paths; the full dtype matrix is acceptance coverage.
        cases += [(torch.bool, "sum"), (torch.uint8, "avg"), (torch.complex64, "sum")]
        with self.communication() as group:
            for dtype, op in cases:
                original = torch.full((4, 6), self.rank + 1, dtype=dtype, device=self.device)
                reference = funcol.wait_tensor(funcol.all_reduce(original.clone(), op, group))
                send = original.clone()  # Caller, not the new API, protects original data.
                result = functional_all_reduce(send, op.upper(), group)
                actual = result + torch.zeros_like(result)  # Exercise automatic waiting.
                output = funcol.wait_tensor(result)
                assert_close(actual, reference)
                self.assertEqual(output.shape, send.shape)
                self.assertEqual(output.dtype, send.dtype)
                self.assertEqual(output.device, send.device)
                self.assertNotEqual(output.data_ptr(), send.data_ptr())
                self.assertTrue(torch.equal(original, torch.full_like(original, self.rank + 1)))

    def test_complex_views_and_rejections(self):
        values = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        with self.communication() as group:
            original = torch.complex(values + self.rank, values + 1).to(self.device)
            for source in (original, original.t()):
                for op in ("sum", "avg"):
                    reference = funcol.wait_tensor(funcol.all_reduce(source.clone(), op, group))
                    send = source.clone()
                    result = funcol.wait_tensor(functional_all_reduce(send, op, group))
                    assert_close(result, reference)
                    self.assertEqual((result.shape, result.dtype), (source.shape, source.dtype))
                    self.assertTrue(result.is_contiguous())
                    self.assertFalse(torch._C._is_alias_of(send, result))
            for op in ("min", "max"):
                with self.assertRaisesRegex(RuntimeError, "complex input only supports SUM/AVG"):
                    functional_all_reduce(original, op, group)
            conjugated = original.conj()
            self.assertTrue(conjugated.is_conj())
            for collective in (funcol.all_reduce, functional_all_reduce):
                with self.assertRaisesRegex(RuntimeError, "unresolved conjugated"):
                    collective(conjugated, "sum", group)

    def test_premul_matches_original_hccl(self):
        with self.communication() as group:
            for dtype in (torch.float16, torch.float32, torch.bfloat16):
                source = torch.full((8,), self.rank + 1, dtype=dtype, device=self.device)
                for factor in (0, -0.5, 2.0):
                    reference = premul_reference(source, factor, group)
                    send = source.clone()
                    result = funcol.wait_tensor(functional_all_reduce(
                        send, "premul_sum", group, premul_factor=factor))
                    assert_close(result, reference)
                    self.assertTrue(torch.equal(send, source))
                    self.assertFalse(torch._C._is_alias_of(send, result))

    def test_premul_rejections(self):
        with self.communication() as group:
            x = torch.ones(8, device=self.device)
            with self.assertRaisesRegex(RuntimeError, "premul_sum requires premul_factor"):
                functional_all_reduce(x, "premul_sum", group)
            with self.assertRaisesRegex(RuntimeError, "premul_factor is only valid for premul_sum"):
                functional_all_reduce(x, "sum", group, premul_factor=0.5)
            for dtype in (torch.bool, torch.uint8, torch.int16, torch.int32):
                send = torch.ones(8, device=self.device, dtype=dtype)
                with self.assertRaisesRegex(RuntimeError, "PreMulSum Data type must be"):
                    functional_all_reduce(send, "premul_sum", group, premul_factor=0.5)
                self.assertTrue(torch.equal(send, torch.ones_like(send)))
            with self.assertRaisesRegex(RuntimeError, "complex input only supports SUM/AVG"):
                functional_all_reduce(x.to(torch.complex64), "premul_sum", group, premul_factor=0.5)

    def test_conversion_source_lifetime(self):
        with self.communication() as group:
            for dtype, op in ((torch.bool, "sum"), (torch.uint8, "avg"), (torch.float32, "premul_sum")):
                factor = -0.5 if op == "premul_sum" else None
                values = (torch.arange(64 * 128).reshape(64, 128) + self.rank) % 7
                values = (values == 0) if dtype == torch.bool else values.to(dtype)
                for use_dlpack in (False, True):
                    if use_dlpack:
                        # Keep no separate owning reference to the wrapped NPU input.
                        send = from_dlpack(to_dlpack(values.to(self.device)))
                    else:
                        # Input preparation must create an internal contiguous source.
                        send = values.to(self.device).t()
                        self.assertFalse(send.is_contiguous())
                    reference = (premul_reference(send, factor, group) if factor is not None else
                                 funcol.wait_tensor(funcol.all_reduce(send.clone(), op, group)))
                    shape, pointer = send.shape, send.data_ptr()
                    pending = functional_all_reduce(send, op, group, premul_factor=factor)
                    del send
                    # Exercise allocator reuse before waiting, without mutating a live input.
                    for _ in range(8):
                        scratch = torch.empty(64 * 128, dtype=dtype, device=self.device)
                        scratch.zero_()
                        del scratch
                    output = funcol.wait_tensor(pending)
                    assert_close(output, reference)
                    self.assertEqual(output.shape, shape)
                    self.assertEqual(output.dtype, dtype)
                    self.assertTrue(output.is_contiguous())
                    self.assertNotEqual(output.data_ptr(), pointer)

    def test_sum_backward(self):
        with self.communication() as group:
            x = torch.full((8,), self.rank + 1.0, device=self.device, requires_grad=True)
            result = functional_all_reduce(x, "sum", group)
            (result * (self.rank + 1)).sum().backward()
            expected = torch.full_like(x, self.world_size * (self.world_size + 1) / 2)
            assert_close(x.grad, expected)

    def test_wrapper_rejection_and_standard_entries(self):
        if not dist.is_gloo_available():
            self.skipTest("Requires Gloo to construct ProcessGroupWrapper")
        from torch._C._distributed_c10d import _ProcessGroupWrapper

        with self.communication(debug_level=dist.DebugLevel.DETAIL) as group:
            self.assertIsInstance(group._get_backend(self.device), _ProcessGroupWrapper)
            x = torch.full((8,), self.rank + 1.0, device=self.device)
            expected = torch.full_like(x, self.world_size * (self.world_size + 1) / 2)
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "requires an unwrapped ProcessGroupHCCL"):
                    functional_all_reduce(x, "sum", group)
                assert_close(funcol.wait_tensor(funcol.all_reduce(x.clone(), "sum", group)), expected)
                inplace = x.clone()
                dist.all_reduce(inplace, group=group)
                assert_close(inplace, expected)

    def test_rejection_and_standard_entry_isolation(self):
        with self.communication() as group:
            x = torch.full((8,), self.rank + 1.0, device=self.device)
            expected = torch.full_like(x, self.world_size * (self.world_size + 1) / 2)
            for _ in range(2):
                standard = funcol.wait_tensor(funcol.all_reduce(x.clone(), "sum", group))
                inplace = x.clone()
                self.assertIsNone(dist.all_reduce(inplace, group=group))
                assert_close(standard, expected)
                assert_close(inplace, expected)
                with self.assertRaisesRegex(RuntimeError, "reduction 'product'.*CCU_MS support matrix"):
                    functional_all_reduce(x, "product", group)
                unsupported = torch.empty(8, dtype=torch.float64, device=self.device)
                with self.assertRaisesRegex(RuntimeError, "actual communication dtype is outside"):
                    functional_all_reduce(unsupported, "sum", group)
                assert_close(funcol.wait_tensor(functional_all_reduce(x.clone(), "sum", group)), expected)
            # Both standard entries still work after the last new call.
            assert_close(funcol.wait_tensor(funcol.all_reduce(x.clone(), "sum", group)), expected)
            dist.all_reduce(x, group=group)
            assert_close(x, expected)

    def test_configuration_rejection(self):
        with self.communication() as group:
            x = torch.ones(8, device=self.device)
            for changes, diagnostic in (
                ({"HCCL_OP_EXPANSION_MODE": "AI_CPU"}, "requires HCCL_OP_EXPANSION_MODE=CCU_MS"),
                ({"HCCL_DETERMINISTIC": "strict"}, "HCCL_DETERMINISTIC=strict"),
            ):
                with patch.dict(os.environ, changes):
                    with self.assertRaisesRegex(RuntimeError, diagnostic):
                        functional_all_reduce(x, "sum", group)
            expected = torch.full_like(x, self.world_size)
            assert_close(funcol.wait_tensor(functional_all_reduce(x.clone(), "sum", group)), expected)


class TestFunctionalAllReduceNonA5(NpuCollectiveTestCase):
    requires_a5 = False

    @property
    def world_size(self):
        return 1

    def test_device_rejection_including_empty(self):
        with self.communication() as group:
            for size in (0, 8):
                x = torch.empty(size, device=self.device)
                with self.assertRaisesRegex(RuntimeError, r"requires Ascend950 \(A5\)"):
                    functional_all_reduce(x, "sum", group)


if __name__ == "__main__":
    run_tests()
