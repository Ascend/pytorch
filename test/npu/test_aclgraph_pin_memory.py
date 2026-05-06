"""
Feature: ACLGraph supports pin_memory during graph capture.
Description: Test that pin_memory() works correctly when called inside an
             ACLGraph capture region.
Expectation: Capture-time pinned blocks are isolated from the default host
             pool, replays see original captured data.
"""

import gc
import os
import subprocess
import sys
import textwrap

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestAclgraphPinMemory(TestCase):

    def setUp(self):
        super().setUp()
        torch.npu.set_device(0)
        gc.collect()
        torch_npu.npu.host_empty_cache()
        torch_npu.npu.reset_peak_host_memory_stats()
        torch_npu.npu.reset_accumulated_host_memory_stats()

    def tearDown(self):
        gc.collect()
        torch_npu.npu.host_empty_cache()
        super().tearDown()

    @staticmethod
    def _warmup(stream):
        with torch.npu.stream(stream):
            _ = torch.ones(64, device="npu") * 2
        torch.npu.synchronize()

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    @torch.no_grad()
    def test_pin_memory_h2d_during_capture(self):
        """End-to-end: capture H2D from a pinned tensor, replay must reproduce data."""
        s = torch.npu.Stream()
        self._warmup(s)

        g = torch.npu.NPUGraph()
        gpu_dst = torch.zeros(64, device="npu")
        with torch.npu.graph(g, stream=s):
            pinned = torch.full((64,), 7.0).pin_memory()
            gpu_dst.copy_(pinned, non_blocking=True)

        self.assertTrue(pinned.is_pinned())

        g.replay()
        torch.npu.synchronize()
        self.assertEqual(gpu_dst.cpu(), torch.full((64,), 7.0))

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    @torch.no_grad()
    def test_captured_pinned_block_isolated_from_default_pool(self):
        """After del of a capture-time pinned tensor, the block must stay in the
        private pool (host_empty_cache cannot free it) and must not be handed
        out to post-capture pin_memory() allocations. Replay must keep reading
        the original captured data."""
        s = torch.npu.Stream()
        self._warmup(s)

        g = torch.npu.NPUGraph()
        gpu_dst = torch.zeros(64, device="npu")
        with torch.npu.graph(g, stream=s):
            pinned = torch.full((64,), 7.0).pin_memory()
            gpu_dst.copy_(pinned, non_blocking=True)

        g.replay()
        torch.npu.synchronize()
        self.assertEqual(gpu_dst.cpu(), torch.full((64,), 7.0))

        captured_ptr = pinned.data_ptr()
        self.assertNotEqual(captured_ptr, 0)
        del pinned
        gc.collect()

        # Key discriminator: host_empty_cache() only drains the default pool.
        # If the captured block landed in the default pool (no private pool
        # isolation), allocations.current will drop. If it landed in the
        # private pool, allocations.current stays the same.
        before = torch_npu.npu.host_memory_stats()
        torch_npu.npu.host_empty_cache()
        after = torch_npu.npu.host_memory_stats()
        self.assertEqual(
            before["allocations.current"], after["allocations.current"],
            "captured pinned block was freed by host_empty_cache — private pool "
            "isolation is broken",
        )
        self.assertEqual(
            before["allocated_bytes.current"], after["allocated_bytes.current"],
        )

        # Post-capture pin_memory() allocations must not reuse the captured address.
        scramble = [torch.full((64,), float(-i - 1)).pin_memory()
                    for i in range(32)]
        self.assertNotIn(
            captured_ptr, {t.data_ptr() for t in scramble},
            "captured pinned block was reused by post-capture pin_memory()",
        )

        # Replay multiple times: data must stay intact.
        for _ in range(5):
            g.replay()
            torch.npu.synchronize()
            self.assertEqual(
                gpu_dst.cpu(), torch.full((64,), 7.0),
                "replay read corrupted data — captured block was recycled",
            )

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    @torch.no_grad()
    def test_intra_capture_block_reuse_stays_in_private_pool(self):
        """Inside a single capture, allocate/del/allocate of pinned blocks must
        work and both captured H2D copies must replay correctly."""
        s = torch.npu.Stream()
        self._warmup(s)

        g = torch.npu.NPUGraph()
        dst1 = torch.zeros(64, device="npu")
        dst2 = torch.zeros(64, device="npu")
        with torch.npu.graph(g, stream=s):
            t1 = torch.full((64,), 3.0).pin_memory()
            dst1.copy_(t1, non_blocking=True)
            del t1
            t2 = torch.full((64,), 5.0).pin_memory()
            dst2.copy_(t2, non_blocking=True)

        self.assertTrue(t2.is_pinned())
        t2_ptr = t2.data_ptr()
        del t2
        gc.collect()

        post = [torch.full((64,), -1.0).pin_memory() for _ in range(16)]
        self.assertNotIn(t2_ptr, {p.data_ptr() for p in post})

        g.replay()
        torch.npu.synchronize()
        self.assertEqual(dst1.cpu(), torch.full((64,), 3.0))
        self.assertEqual(dst2.cpu(), torch.full((64,), 5.0))

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    @torch.no_grad()
    def test_two_graphs_shared_pool_no_cross_contamination(self):
        """Two graphs sharing a pool: each captured H2D must replay its own data."""
        s = torch.npu.Stream()
        self._warmup(s)

        pool = torch.npu.graph_pool_handle()
        g1 = torch.npu.NPUGraph()
        g2 = torch.npu.NPUGraph()
        d1 = torch.zeros(64, device="npu")
        d2 = torch.zeros(64, device="npu")

        with torch.npu.graph(g1, stream=s, pool=pool):
            t1 = torch.full((64,), 11.0).pin_memory()
            d1.copy_(t1, non_blocking=True)
        with torch.npu.graph(g2, stream=s, pool=pool):
            t2 = torch.full((64,), 22.0).pin_memory()
            d2.copy_(t2, non_blocking=True)

        self.assertTrue(t1.is_pinned())
        self.assertTrue(t2.is_pinned())
        self.assertNotEqual(t1.data_ptr(), t2.data_ptr())

        # Interleave replays multiple times.
        for _ in range(3):
            g2.replay()
            g1.replay()
            g1.replay()
            g2.replay()
        torch.npu.synchronize()
        self.assertEqual(d1.cpu(), torch.full((64,), 11.0))
        self.assertEqual(d2.cpu(), torch.full((64,), 22.0))

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    @torch.no_grad()
    def test_release_pool_frees_unused_captured_blocks(self):
        """Capture-time pinned blocks that were never used by any captured op
        must be reclaimable after g.reset() + host_empty_cache().
        """
        s = torch.npu.Stream()
        self._warmup(s)

        gc.collect()
        torch_npu.npu.host_empty_cache()
        baseline = torch_npu.npu.host_memory_stats()

        N = 10
        for _ in range(N):
            g = torch.npu.NPUGraph()
            with torch.npu.graph(g, stream=s):
                # Allocated inside capture but never referenced by a captured
                # op — this is the branch the base allocator promises to
                # reclaim on release_pool + empty_cache.
                t = torch.full((1024,), 1.0).pin_memory()
            del t
            g.reset()
            del g

        gc.collect()
        torch_npu.npu.host_empty_cache()
        final = torch_npu.npu.host_memory_stats()

        self.assertEqual(
            final["allocations.current"], baseline["allocations.current"],
            f"unused captured pinned blocks leaked "
            f"{final['allocations.current'] - baseline['allocations.current']} "
            f"over {N} capture/reset cycles — release_pool did not hand blocks "
            f"back to the default pool, or empty_cache did not drain them",
        )
        self.assertEqual(
            final["allocated_bytes.current"],
            baseline["allocated_bytes.current"],
        )

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_capture_rejects_expandable_segments(self):
        """pin_memory_expandable_segments=True must be rejected at capture_begin.
        Runs in a subprocess because setAllocator() uses std::call_once and
        cannot be reconfigured within the current process. Requires CANN
        driver >= 25.5.0 to actually enable the expandable segments path;
        on older drivers the setting is silently downgraded and this path
        cannot be exercised."""
        probe = textwrap.dedent(
            """
            import os
            os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'pin_memory_expandable_segments:True'
            import torch, torch_npu
            torch.npu.set_device(0)
            # Touch allocator so config parsing runs and any driver downgrade
            # warning is emitted before we attempt capture.
            _ = torch.empty(1, device='npu')
            g = torch.npu.NPUGraph()
            s = torch.npu.Stream()
            try:
                with torch.npu.graph(g, stream=s):
                    pass
            except RuntimeError as e:
                msg = str(e)
                if 'pin_memory_expandable_segments' in msg:
                    print('OK_REJECT')
                else:
                    print('OTHER_ERROR:' + msg)
            else:
                print('NO_ERROR_RAISED')
            """
        ).strip()
        env = os.environ.copy()
        env['PYTORCH_NPU_ALLOC_CONF'] = 'pin_memory_expandable_segments:True'
        r = subprocess.run(
            [sys.executable, '-c', probe],
            capture_output=True, text=True, timeout=180, env=env,
        )
        combined = (r.stdout or '') + (r.stderr or '')
        driver_downgraded = (
            'pin_memory_expandable_segments setting failure' in combined
            or 'does not support this feature' in combined
        )
        if driver_downgraded and 'OK_REJECT' not in r.stdout:
            self.skipTest(
                "CANN driver does not support pin_memory_expandable_segments "
                "(needs >= 25.5.0); cannot exercise capture_begin rejection path"
            )
        self.assertIn('OK_REJECT', r.stdout,
                      msg=f"stdout={r.stdout!r}\nstderr={r.stderr!r}")


if __name__ == "__main__":
    run_tests()
