"""Regression tests for the PR 47558 backport on v2.9.0_a5_copy_merge.

This branch retains NZ -> CPU support via an explicit base-format cast.
Common copy checks also run on Ascend910B; rejection checks require Ascend950.
References are constructed directly with torch.arange.
Value checks cast to ND explicitly to avoid depending on the NZ -> CPU path.

Run in separate processes to cover both host dispatch paths:
    TASK_QUEUE_ENABLE=0 python test/npu/test_copy_nz_aclnn.py
    TASK_QUEUE_ENABLE=1 python test/npu/test_copy_nz_aclnn.py
"""

import unittest
from itertools import product

from parameterized import parameterized

import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices


ND = 2
NZ = 29
COPY_ERROR = "The copy_ operator with internal format tensors is not supported on Ascend950"


class TestCopyNzAclnn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU is required")
        if not torch.npu.get_device_name().startswith(("Ascend910B", "Ascend950")):
            raise unittest.SkipTest("These copy checks require Ascend910B or Ascend950")
        super().setUpClass()

    def setUp(self):
        super().setUp()
        # _npuConfig exposes a setter only; read the backing option instead.
        option = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
        old_allow_internal_format = option is not None and option.decode() == "enable"
        self.addCleanup(
            setattr, torch.npu.config, "allow_internal_format", old_allow_internal_format
        )
        torch.npu.config.allow_internal_format = True
        self.device = torch.device("npu", torch.npu.current_device())

    @staticmethod
    def _metadata(tensor):
        return (
            tensor.shape, tensor.stride(), tensor.dtype, tensor.device,
            tensor.storage_offset(), tensor.data_ptr(),
            tensor.untyped_storage().nbytes(),
            torch_npu.get_npu_format(tensor),
        )

    @parameterized.expand(product(
        (torch.float16, torch.float32), ((32, 64), (17, 33)), (False, True)
    ))
    def test_same_device_copy(self, dtype, shape, non_blocking):
        expected = torch.arange(shape[0] * shape[1]).reshape(shape).to(dtype)
        src = torch_npu.npu_format_cast(expected.to(self.device), NZ)
        dst = torch_npu.npu_format_cast(torch.full_like(expected, -100).to(self.device), NZ)
        before = self._metadata(dst)
        if shape == (17, 33):
            self.assertGreater(
                dst.untyped_storage().nbytes(),
                dst.numel() * dst.element_size(),
                "The padding case must allocate more than logical tensor bytes",
            )
        self.assertIs(dst.copy_(src, non_blocking=non_blocking), dst)
        torch.npu.synchronize(self.device)
        self.assertEqual(self._metadata(dst), before)
        self.assertTrue(torch.equal(torch_npu.npu_format_cast(dst, ND).cpu(), expected))
        self.assertTrue(torch.equal(torch_npu.npu_format_cast(src, ND).cpu(), expected))

    def test_self_copy_and_repeated_overwrite(self):
        expected = torch.arange(17 * 33).reshape(17, 33).to(torch.float16)
        dst = torch_npu.npu_format_cast(expected.to(self.device), NZ)
        self.assertIs(dst.copy_(dst), dst)
        self.assertTrue(torch.equal(torch_npu.npu_format_cast(dst, ND).cpu(), expected))
        references = [
            expected,
            torch.arange(17 * 33, 0, -1).reshape(17, 33).to(torch.float16),
        ]
        sources = [torch_npu.npu_format_cast(reference.to(self.device), NZ) for reference in references]
        for src in sources * 2:
            dst.copy_(src, non_blocking=True)
        torch.npu.synchronize(self.device)
        self.assertTrue(torch.equal(torch_npu.npu_format_cast(dst, ND).cpu(), references[-1]))

    @parameterized.expand([
        ("dtype_mismatch",),
        ("broadcast",),
        ("nd_to_nz",),
        ("nz_to_nd",),
        ("host_to_nz",),
    ])
    @SupportedDevices(["Ascend950"])
    def test_reject_incompatible_copy(self, case):
        expected = torch.arange(32 * 64).reshape(32, 64).to(torch.float16)
        dst = torch_npu.npu_format_cast(torch.full_like(expected, -100).to(self.device), NZ)
        src = torch_npu.npu_format_cast(expected.to(self.device), NZ)
        if case == "dtype_mismatch":
            src = torch_npu.npu_format_cast(expected.to(self.device, dtype=torch.float32), NZ)
        elif case == "broadcast":
            src = torch_npu.npu_format_cast(expected[:16].to(self.device), NZ)
        elif case == "nd_to_nz":
            src = torch_npu.npu_format_cast(expected.to(self.device), ND)
        elif case == "nz_to_nd":
            dst = torch_npu.npu_format_cast(torch.full_like(expected, -100).to(self.device), ND)
        elif case == "host_to_nz":
            src = expected
        before = torch_npu.npu_format_cast(dst, ND).cpu().clone()
        with self.assertRaisesRegex(RuntimeError, COPY_ERROR):
            dst.copy_(src)
        self.assertTrue(torch.equal(torch_npu.npu_format_cast(dst, ND).cpu(), before))

    @parameterized.expand([(False,), (True,)])
    @SupportedDevices(["Ascend950"])
    def test_reject_views(self, view_is_source):
        expected = torch.arange(32 * 64).reshape(32, 64).to(torch.float16)
        base = torch_npu.npu_format_cast(expected.to(self.device), NZ)
        view = base.view_as(base)
        # A contiguous, zero-offset view must still be rejected.
        self.assertTrue(view._is_view())
        self.assertTrue(view.is_contiguous())
        self.assertEqual(view.storage_offset(), 0)
        self.assertEqual(torch_npu.get_npu_format(view), NZ)
        peer = torch_npu.npu_format_cast(torch.full_like(expected, -100).to(self.device), NZ)
        dst, src = (peer, view) if view_is_source else (view, peer)
        observed = peer if view_is_source else base
        before = torch_npu.npu_format_cast(observed, ND).cpu().clone()
        with self.assertRaisesRegex(RuntimeError, COPY_ERROR):
            dst.copy_(src)
        self.assertTrue(torch.equal(torch_npu.npu_format_cast(observed, ND).cpu(), before))

    @SupportedDevices(["Ascend950"])
    def test_reject_detached_slices(self):
        expected = torch.ones((32, 64), dtype=torch.float16)
        dst_base = torch_npu.npu_format_cast(expected.to(self.device), NZ)
        src_base = torch_npu.npu_format_cast((expected * 2).to(self.device), NZ)
        dst, src = dst_base[:16].detach(), src_base[:16].detach()
        # detach clears the view flag but retains the full backing storage.
        for tensor in (dst, src):
            self.assertFalse(tensor._is_view())
            self.assertTrue(tensor.is_contiguous())
            self.assertEqual(tensor.storage_offset(), 0)
            self.assertEqual(torch_npu.get_npu_format(tensor), NZ)
        with self.assertRaisesRegex(RuntimeError, COPY_ERROR):
            dst.copy_(src)
        torch.npu.synchronize(self.device)
        self.assertTrue(torch.equal(torch_npu.npu_format_cast(dst_base, ND).cpu(), expected))

    @parameterized.expand(product((torch.float16, torch.float32), (False, True)))
    def test_nz_to_host(self, dtype, non_blocking):
        expected = torch.arange(17 * 33).reshape(17, 33).to(dtype)
        src = torch_npu.npu_format_cast(expected.to(self.device), NZ)
        before = self._metadata(src)
        host = torch.full_like(expected, -100)
        self.assertIs(host.copy_(src, non_blocking=non_blocking), host)
        torch.npu.synchronize(self.device)
        self.assertTrue(torch.equal(host, expected))
        self.assertEqual(self._metadata(src), before)
        self.assertTrue(torch.equal(torch_npu.npu_format_cast(src, ND).cpu(), expected))

    @SupportedDevices(["Ascend950"])
    def test_reject_cross_device(self):
        if torch.npu.device_count() < 2:
            self.skipTest("Two NPUs are required for cross-device rejection")
        expected = torch.arange(32 * 64).reshape(32, 64).to(torch.float16)
        other_device = torch.device("npu", (self.device.index + 1) % torch.npu.device_count())
        src = torch_npu.npu_format_cast(expected.to(other_device), NZ)
        dst = torch_npu.npu_format_cast(torch.full_like(expected, -100).to(self.device), NZ)
        before = torch_npu.npu_format_cast(dst, ND).cpu().clone()
        with self.assertRaisesRegex(RuntimeError, COPY_ERROR):
            dst.copy_(src)
        self.assertTrue(torch.equal(torch_npu.npu_format_cast(dst, ND).cpu(), before))

    @parameterized.expand([(False,), (True,)])
    def test_custom_stream_ordering(self, non_blocking):
        expected = torch.arange(17 * 33).reshape(17, 33).to(torch.float16)
        src_nd = expected.to(self.device)
        dst = torch_npu.npu_format_cast(torch.full_like(expected, -100).to(self.device), NZ)
        torch.npu.synchronize(self.device)
        stream = torch.npu.Stream(device=self.device)
        with torch.npu.stream(stream):
            # Producer, memcpy and consumer must all use this stream.
            src = torch_npu.npu_format_cast(src_nd, NZ)
            self.assertEqual(torch_npu.get_npu_format(src), NZ)
            self.assertIs(dst.copy_(src, non_blocking=non_blocking), dst)
            result_nd = torch_npu.npu_format_cast(dst, ND)
        stream.synchronize()
        self.assertTrue(torch.equal(result_nd.cpu(), expected))

if __name__ == "__main__":
    unittest.main()
