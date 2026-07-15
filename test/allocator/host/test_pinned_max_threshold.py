"""Tests for pinned_max_round_threshold_mb and pinned_max_cached_size_mb config options.

These options control pinned memory allocator behavior on the default
(non-expandable) host allocator:
- pinned_max_round_threshold_mb: Allocations above this threshold skip power-of-2
  rounding and use exact size instead.
- pinned_max_cached_size_mb: Freed blocks above this threshold are released to the
  OS immediately instead of being cached in the free list.

When pin_memory_expandable_segments is enabled, the expandable host allocator path
does not consult these thresholds. Setting either option together with
pin_memory_expandable_segments emits a one-time warning and the thresholds stay
inert (process still starts normally); see TestPinnedThresholdExpandableHost.

Most tests run in-process: pinned_max_* are overwrite-assign (not min), so each
test setting its own threshold does not interfere with others, and each test
clears the host cache on entry/exit. Env-var validation and
pin_memory_expandable_segments tests use a subprocess to get a clean
PYTORCH_NPU_ALLOC_CONF context per case (env var reads are sticky --
std::call_once in NPUAllocatorConfig::instance() only parses once per process)
and to assert on the one-time warning that parseArgs emits.
"""

import gc
import os
import subprocess
import sys
import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.npu.memory import _set_allocator_settings
from torch_npu.npu.utils import get_cann_version, _is_gte_cann_version


ALLOCATED_BYTES_CURRENT = "allocated_bytes.current"
ACTIVE_BYTES_CURRENT = "active_bytes.current"


def _expandable_runtime_supported():
    """pin_memory_expandable_segments requires CANN >= 8.5.0 and driver >= 25.5.0
    (see NPUAllocatorConfig.cpp: pinMemoryExpandableMinCannVersion /
    pinMemoryExpandableMinDriverVersion). The warning-only expandable tests are
    skipped when the runtime would itself downgrade the option back to False,
    because then m_pin_memory_expandable_segments never becomes true and the
    warning never fires.

    Returns False on any probe error; those tests will be skipped rather than
    crash the suite. The skip is visible in pytest's skip count.
    """
    try:
        if not _is_gte_cann_version("8.5.0", "CANN"):
            return False
        driver_version = get_cann_version(module="DRIVER")
        parts = [int(x) for x in driver_version.split('.') if x.isdigit()]
        if len(parts) < 2:
            return False
        major, minor = parts[0], parts[1]
        if major > 25:
            return True
        if major < 25:
            return False
        return minor >= 5
    except Exception:
        return False


_SUBPROCESS_SCRIPT = (
    "import torch\n"
    "torch.empty(1024, dtype=torch.uint8, device='cpu').pin_memory()\n"
    "print('ok')\n"
)


def _run_subprocess(env_conf: str, script: str = _SUBPROCESS_SCRIPT,
                    timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a child python process with PYTORCH_NPU_ALLOC_CONF set. The default
    script allocates a pinned tensor to trigger NPUAllocatorConfig::instance()
    call_once -> parseArgs(env_str). On a parse error the child raises
    (non-zero returncode); the one-time warning that parseArgs emits on the
    expandable+threshold combo lands in stderr. Pass `script` to drive
    env-var-only behavior assertions from the parent."""
    env = os.environ.copy()
    env["PYTORCH_NPU_ALLOC_CONF"] = env_conf
    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestPinnedMaxRoundThresholdCachingHost(TestCase):
    """Test pinned_max_round_threshold_mb with the default (non-expandable) host allocator."""

    def test_round_threshold_skip_rounding(self):
        """Allocations above pinned_max_round_threshold_mb skip power-of-2
        rounding and use the exact requested size."""
        _set_allocator_settings("pinned_max_round_threshold_mb:64")
        gc.collect()
        torch_npu.npu.host_empty_cache()

        # 80 MB > 64 MB threshold -> no power-of-2 rounding (would be 128 MB).
        t = torch.empty(80 * 1024 * 1024, dtype=torch.uint8, device="cpu").pin_memory()
        stats = torch_npu.npu.host_memory_stats()
        # active_bytes reflects the block currently checked out to us.
        self.assertGreaterEqual(stats[ACTIVE_BYTES_CURRENT] / (1024 * 1024), 80)
        # allocated_bytes (= active + cached) should be ~80 MB, not 128 MB.
        allocated_mb = stats[ALLOCATED_BYTES_CURRENT] / (1024 * 1024)
        self.assertLess(allocated_mb, 85, f"Expected ~80 MB, got {allocated_mb} MB")
        self.assertGreaterEqual(allocated_mb, 80, f"Expected >=80 MB, got {allocated_mb} MB")

        t = None
        gc.collect()
        torch_npu.npu.host_empty_cache()

    def test_round_threshold_below_threshold_still_rounded(self):
        """Allocations below pinned_max_round_threshold_mb still get power-of-2
        rounding. 10 MB with threshold=128 rounds up to 16 MB."""
        _set_allocator_settings("pinned_max_round_threshold_mb:128")
        gc.collect()
        torch_npu.npu.host_empty_cache()

        t = torch.empty(10 * 1024 * 1024, dtype=torch.uint8, device="cpu").pin_memory()
        stats = torch_npu.npu.host_memory_stats()
        # active_bytes reflects the block currently checked out (>= 10 MB).
        self.assertGreaterEqual(stats[ACTIVE_BYTES_CURRENT] / (1024 * 1024), 10)
        allocated_mb = stats[ALLOCATED_BYTES_CURRENT] / (1024 * 1024)
        # PowerOf2Ceil(10 MiB) = 16 MiB.
        self.assertGreaterEqual(allocated_mb, 16, f"Expected >=16 MB (rounded), got {allocated_mb} MB")
        # Should not round to next power of two (32 MB).
        self.assertLess(allocated_mb, 32, f"Expected <32 MB, got {allocated_mb} MB")

        t = None
        gc.collect()
        torch_npu.npu.host_empty_cache()


class TestPinnedMaxCachedSizeCachingHost(TestCase):
    """Test pinned_max_cached_size_mb with the default (non-expandable) host allocator."""

    def test_cached_size_blocks_released(self):
        """Blocks above pinned_max_cached_size_mb are released on free, not
        cached. After freeing + empty_cache, allocated_bytes drops to 0."""
        _set_allocator_settings("pinned_max_cached_size_mb:32")
        gc.collect()
        torch_npu.npu.host_empty_cache()

        t = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cpu").pin_memory()
        stats_after_alloc = torch_npu.npu.host_memory_stats()
        # Verify the allocation actually succeeded (active_bytes >= 64 MB).
        self.assertGreaterEqual(
            stats_after_alloc[ACTIVE_BYTES_CURRENT] / (1024 * 1024), 64,
            f"Expected active >=64 MB after alloc, got {stats_after_alloc[ACTIVE_BYTES_CURRENT]}")

        t = None
        gc.collect()
        # After free (before empty_cache), block > 32 MB threshold is released
        # immediately, so allocated_bytes should already be 0.
        self.assertEqual(torch_npu.npu.host_memory_stats()[ALLOCATED_BYTES_CURRENT], 0,
                         "Block > pinned_max_cached_size_mb should be released on free, not cached")
        # empty_cache is a no-op here, but call it for symmetry with other tests.
        torch_npu.npu.host_empty_cache()
        self.assertEqual(torch_npu.npu.host_memory_stats()[ALLOCATED_BYTES_CURRENT], 0)

    def test_cached_size_blocks_below_threshold_cached(self):
        """Blocks below pinned_max_cached_size_mb stay cached on free. After
        free (without empty_cache), allocated_bytes stays > 0 (cached) while
        active_bytes drops to 0 (no longer checked out); after empty_cache,
        allocated_bytes drops to 0."""
        _set_allocator_settings("pinned_max_cached_size_mb:128")
        gc.collect()
        torch_npu.npu.host_empty_cache()

        t = torch.empty(10 * 1024 * 1024, dtype=torch.uint8, device="cpu").pin_memory()
        t = None
        gc.collect()
        # 10 MB < 128 MB threshold, so block is cached.
        stats_before_empty = torch_npu.npu.host_memory_stats()
        self.assertEqual(stats_before_empty[ACTIVE_BYTES_CURRENT], 0,
                         "active_bytes should be 0 after free (no longer checked out)")
        self.assertGreater(stats_before_empty[ALLOCATED_BYTES_CURRENT], 0,
                           "Block < pinned_max_cached_size_mb should stay cached after free")
        torch_npu.npu.host_empty_cache()
        self.assertEqual(torch_npu.npu.host_memory_stats()[ALLOCATED_BYTES_CURRENT], 0)

    def test_env_var_pinned_max_round_threshold(self):
        """pinned_max_round_threshold_mb via PYTORCH_NPU_ALLOC_CONF drives
        allocator behavior (no _set_allocator_settings call in the child).
        80 MB > 64 MB threshold -> no power-of-2 rounding (would be 128 MB)."""
        script = (
            "import gc, torch, torch_npu\n"
            "t = torch.empty(80 * 1024 * 1024, dtype=torch.uint8, device='cpu').pin_memory()\n"
            "print(torch_npu.npu.host_memory_stats()['allocated_bytes.current'])\n"
        )
        result = _run_subprocess("pinned_max_round_threshold_mb:64", script=script)
        self.assertEqual(result.returncode, 0,
                         f"Process failed: stdout={result.stdout}\nstderr={result.stderr}")
        allocated_mb = int(result.stdout.strip()) / (1024 * 1024)
        self.assertLess(allocated_mb, 128, f"Expected <128 MB (no rounding), got {allocated_mb} MB")
        self.assertGreaterEqual(allocated_mb, 80, f"Expected >=80 MB, got {allocated_mb} MB")

    def test_env_var_pinned_max_cached_size(self):
        """pinned_max_cached_size_mb via PYTORCH_NPU_ALLOC_CONF drives
        allocator behavior (no _set_allocator_settings call in the child).
        64 MB > 32 MB threshold -> block released on free, not cached."""
        script = (
            "import gc, torch, torch_npu\n"
            "t = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device='cpu').pin_memory()\n"
            "t = None\n"
            "gc.collect()\n"
            "print(torch_npu.npu.host_memory_stats()['allocated_bytes.current'])\n"
        )
        result = _run_subprocess("pinned_max_cached_size_mb:32", script=script)
        self.assertEqual(result.returncode, 0,
                         f"Process failed: stdout={result.stdout}\nstderr={result.stderr}")
        self.assertEqual(int(result.stdout.strip()), 0,
                         "Block > pinned_max_cached_size_mb should be released on free, not cached")


class TestPinnedThresholdConsistency(TestCase):
    """Test combined threshold behavior."""

    def test_round_threshold_gt_cached_size_skips_rounding(self):
        """When pinned_max_round_threshold_mb > pinned_max_cached_size_mb,
        allocations above pinned_max_cached_size_mb skip rounding regardless of
        pinned_max_round_threshold_mb (cached_size takes precedence)."""
        # round=256, cached=128; 130 MB > cached -> skip rounding, ~130 MB.
        _set_allocator_settings(
            "pinned_max_round_threshold_mb:256,pinned_max_cached_size_mb:128"
        )
        gc.collect()
        torch_npu.npu.host_empty_cache()

        t = torch.empty(130 * 1024 * 1024, dtype=torch.uint8, device="cpu").pin_memory()
        stats = torch_npu.npu.host_memory_stats()
        self.assertGreaterEqual(stats[ACTIVE_BYTES_CURRENT] / (1024 * 1024), 130)
        allocated_mb = stats[ALLOCATED_BYTES_CURRENT] / (1024 * 1024)
        self.assertLess(allocated_mb, 256, f"Expected <256 MB (no rounding), got {allocated_mb} MB")
        self.assertGreaterEqual(allocated_mb, 130, f"Expected >=130 MB, got {allocated_mb} MB")

        t = None
        gc.collect()
        # 130 MB > 128 MB cached threshold -> released immediately, not cached.
        self.assertEqual(torch_npu.npu.host_memory_stats()[ALLOCATED_BYTES_CURRENT], 0,
                         "Block > pinned_max_cached_size_mb should be released on free")
        torch_npu.npu.host_empty_cache()

    def test_both_thresholds_combined(self):
        """Both thresholds set to 64 MB: 80 MB alloc uses exact size (no
        rounding) and is released (not cached) on free."""
        _set_allocator_settings(
            "pinned_max_round_threshold_mb:64,pinned_max_cached_size_mb:64"
        )
        gc.collect()
        torch_npu.npu.host_empty_cache()

        t = torch.empty(80 * 1024 * 1024, dtype=torch.uint8, device="cpu").pin_memory()
        stats_after_alloc = torch_npu.npu.host_memory_stats()
        self.assertGreaterEqual(stats_after_alloc[ACTIVE_BYTES_CURRENT] / (1024 * 1024), 80)
        allocated_mb = stats_after_alloc[ALLOCATED_BYTES_CURRENT] / (1024 * 1024)
        self.assertLess(allocated_mb, 128, f"Expected ~80 MB (no rounding), got {allocated_mb} MB")
        self.assertGreaterEqual(allocated_mb, 80, f"Expected >=80 MB, got {allocated_mb} MB")

        t = None
        gc.collect()
        # 80 MB > 64 MB cached threshold -> released immediately, not cached.
        self.assertEqual(torch_npu.npu.host_memory_stats()[ALLOCATED_BYTES_CURRENT], 0,
                         "Block > pinned_max_cached_size_mb should be released on free")
        torch_npu.npu.host_empty_cache()
        self.assertEqual(torch_npu.npu.host_memory_stats()[ALLOCATED_BYTES_CURRENT], 0)


class TestPinnedThresholdExpandableHost(TestCase):
    """pin_memory_expandable_segments uses a dedicated expandable allocator path
    that does not consult pinned_max_round_threshold_mb or
    pinned_max_cached_size_mb. Setting either option together with
    pin_memory_expandable_segments emits a one-time warning and the thresholds
    stay inert (process still starts normally).

    The warning-only tests are skipped when the runtime would downgrade
    pin_memory_expandable_segments back to False (CANN < 8.5 or driver < 25.5),
    because in that case m_pin_memory_expandable_segments never becomes true and
    the warning never fires. test_expandable_without_threshold_accepted does
    not depend on the warning and runs on all environments.
    """

    @unittest.skipIf(not _expandable_runtime_supported(),
                     "pin_memory_expandable_segments downgraded to False by runtime "
                     "(CANN < 8.5 or driver < 25.5); warning never fires")
    def test_expandable_with_round_threshold_warns(self):
        """pin_memory_expandable_segments:True + pinned_max_round_threshold_mb
        emits a warning and the process starts normally."""
        result = _run_subprocess(
            "pin_memory_expandable_segments:True,pinned_max_round_threshold_mb:64"
        )
        self.assertEqual(result.returncode, 0,
                         f"Expected success with warning. stdout={result.stdout}\nstderr={result.stderr}")
        self.assertIn("ok", result.stdout)
        combined = (result.stderr or "") + (result.stdout or "")
        self.assertIn("pinned_max_round_threshold_mb", combined)
        self.assertIn("pin_memory_expandable_segments", combined)

    @unittest.skipIf(not _expandable_runtime_supported(),
                     "pin_memory_expandable_segments downgraded to False by runtime "
                     "(CANN < 8.5 or driver < 25.5); warning never fires")
    def test_expandable_with_cached_size_warns(self):
        """pin_memory_expandable_segments:True + pinned_max_cached_size_mb
        emits a warning and the process starts normally."""
        result = _run_subprocess(
            "pin_memory_expandable_segments:True,pinned_max_cached_size_mb:32"
        )
        self.assertEqual(result.returncode, 0,
                         f"Expected success with warning. stdout={result.stdout}\nstderr={result.stderr}")
        self.assertIn("ok", result.stdout)
        combined = (result.stderr or "") + (result.stdout or "")
        self.assertIn("pinned_max_cached_size_mb", combined)
        self.assertIn("pin_memory_expandable_segments", combined)

    def test_expandable_without_threshold_accepted(self):
        """pin_memory_expandable_segments:True alone starts normally and emits
        no pinned_max_* threshold warning. On runtimes that downgrade
        pin_memory_expandable_segments to False, the downgrade warning may
        appear but it does not mention the threshold options."""
        result = _run_subprocess("pin_memory_expandable_segments:True")
        self.assertEqual(result.returncode, 0,
                         f"Process failed: stdout={result.stdout}\nstderr={result.stderr}")
        self.assertIn("ok", result.stdout)
        combined = (result.stderr or "") + (result.stdout or "")
        self.assertNotIn("pinned_max_round_threshold_mb", combined)
        self.assertNotIn("pinned_max_cached_size_mb", combined)


if __name__ == "__main__":
    run_tests()
