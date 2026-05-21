# Owner(s): ["module: npu"]
"""
Tests for NPU Sanitizer record_stream detection.

record_stream detection is separate from data race detection:
  - Data race: detected at kernel launch time (raises CUDASanitizerErrors)
  - Missing record_stream: detected at deallocation or via flush_record_stream_warnings()

Per PyTorch docs (torch.Tensor.record_stream), record_stream is NOT needed when
creation_stream has synced with usage_stream before tensor deallocation:
  - creation_stream.wait_stream(usage_stream)
  - creation_stream.wait_event(event recorded on usage_stream)
  - torch_npu.npu.synchronize() (device-level sync covers all directions)

Conversely, usage_stream.wait_stream(creation_stream) only resolves data races
but does NOT guarantee memory safety — record_stream is still needed.

Test matrix:
┌───────────────────────────────────┬──────────────┬───────────────────────┐
│ Scenario                          │ Data race?   │ record_stream needed? │
├───────────────────────────────────┼──────────────┼───────────────────────┤
│ No sync at all                    │ Yes          │ Yes                   │
│ usage.wait(creation) only         │ No           │ Yes                   │
│ creation.wait(usage) only         │ Possible     │ No                    │
│ Both directions synced            │ No           │ No                    │
│ device synchronize                │ No           │ No                    │
│ record_stream called              │ Unresolved   │ No (recorded)         │
│ Same stream                       │ No           │ No                    │
└───────────────────────────────────┴──────────────┴───────────────────────┘
"""

import gc
import os

import torch
import torch.cuda._sanitizer as csan
import torch.distributed as dist
   
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


def setup_sanitizer():
    """Enable sanitizer with record_stream checking."""
    os.environ['TORCH_NPU_SANITIZER'] = '1'
    import torch_npu.npu._sanitizer as sanitizer
    if not sanitizer.npu_sanitizer.enabled:
        sanitizer.npu_sanitizer.enable()


def reset_sanitizer():
    """Reset sanitizer state between tests."""
    import torch_npu.npu._sanitizer as sanitizer
    if sanitizer.npu_sanitizer.dispatch is not None:
        try:
            sanitizer.npu_sanitizer.dispatch.__exit__(None, None, None)
        except Exception:
            pass
        sanitizer.npu_sanitizer.dispatch = None
    sanitizer.npu_sanitizer.event_handler = None
    sanitizer.npu_sanitizer.enabled = False


def get_event_handler():
    import torch_npu.npu._sanitizer as sanitizer
    return sanitizer.npu_sanitizer.event_handler


class SanitizerRecordStreamTestBase(TestCase):
    def setUp(self):
        reset_sanitizer()
        setup_sanitizer()

    def tearDown(self):
        reset_sanitizer()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def _get_tracked_tensor_info(tensor):
        storage_ptr = tensor.untyped_storage().data_ptr()
        info = get_event_handler()._npu_tensors.get(storage_ptr)
        if info is None:
            raise AssertionError(
                f"Tensor storage {storage_ptr} is not tracked by NPU sanitizer."
            )
        return storage_ptr, info


class TestDataRaceDetection(SanitizerRecordStreamTestBase):
    def test_write_read_race(self):
        """Unsynchronized cross-stream read after write should raise data race."""
        x = torch.randn(100, device="npu")
        stream = torch_npu.npu.Stream()

        with self.assertRaises(csan.CUDASanitizerErrors):
            with torch_npu.npu.stream(stream):
                _ = x + 1

    def test_record_stream_does_not_fix_data_race(self):
        """record_stream should not hide a real cross-stream data race."""
        x = torch.randn(100, device="npu")
        stream = torch_npu.npu.Stream()
        x.record_stream(stream)

        with self.assertRaises(csan.CUDASanitizerErrors):
            with torch_npu.npu.stream(stream):
                _ = x + 1


class TestMissingRecordStream(SanitizerRecordStreamTestBase):
    def test_cross_stream_no_record_stream(self):
        """Cross-stream use without record_stream should report missing record_stream."""
        x = torch.randn(100, device="npu")
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        stream.wait_stream(default_stream)
        with torch_npu.npu.stream(stream):
            _ = x + 1

        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertGreater(len(warnings), 0)

    def test_multiple_streams_need_record_stream(self):
        """Each non-creation stream needs its own record_stream coverage."""
        x = torch.randn(100, device="npu")
        stream1 = torch_npu.npu.Stream()
        stream2 = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        stream1.wait_stream(default_stream)
        with torch_npu.npu.stream(stream1):
            _ = x + 1

        stream2.wait_stream(default_stream)
        with torch_npu.npu.stream(stream2):
            _ = x + 2

        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertGreaterEqual(len(warnings), 2)


class TestRecordStreamNotNeeded(SanitizerRecordStreamTestBase):
    def test_record_stream_suppresses_warning(self):
        """record_stream should suppress missing-record_stream warning for that stream."""
        x = torch.randn(100, device="npu")
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        x.record_stream(stream)
        stream.wait_stream(default_stream)
        with torch_npu.npu.stream(stream):
            _ = x + 1

        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertEqual(len(warnings), 0)

    def test_creation_waits_usage_via_wait_stream(self):
        """creation_stream.wait_stream(usage_stream) should make record_stream unnecessary."""
        x = torch.randn(100, device="npu")
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        stream.wait_stream(default_stream)
        with torch_npu.npu.stream(stream):
            _ = x + 1

        default_stream.wait_stream(stream)
        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertEqual(len(warnings), 0)

    def test_creation_waits_usage_via_event(self):
        """creation_stream.wait_event(event_on_usage) should make record_stream unnecessary."""
        x = torch.randn(100, device="npu")
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        stream.wait_stream(default_stream)
        with torch_npu.npu.stream(stream):
            _ = x + 1
            event = torch_npu.npu.Event()
            event.record(stream)

        default_stream.wait_event(event)
        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertEqual(len(warnings), 0)

    def test_device_sync(self):
        """Device synchronize should cover prior cross-stream uses."""
        x = torch.randn(100, device="npu")
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        stream.wait_stream(default_stream)
        with torch_npu.npu.stream(stream):
            _ = x + 1

        torch_npu.npu.synchronize()

        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertEqual(len(warnings), 0)


class TestRecordStreamSequenceBoundaries(SanitizerRecordStreamTestBase):
    def test_creation_waits_usage_only_covers_prior_uses(self):
        """A mid-sequence reverse wait should not cover later cross-stream uses."""
        x = torch.randn(100, device="npu")
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        stream.wait_stream(default_stream)
        with torch_npu.npu.stream(stream):
            _ = x + 1

        default_stream.wait_stream(stream)
        with torch_npu.npu.stream(stream):
            _ = x + 2

        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertGreater(len(warnings), 0)

    def test_creation_wait_event_only_covers_prior_uses(self):
        """A mid-sequence event wait should not cover later cross-stream uses."""
        x = torch.randn(100, device="npu")
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        stream.wait_stream(default_stream)
        with torch_npu.npu.stream(stream):
            _ = x + 1
            event = torch_npu.npu.Event()
            event.record(stream)

        default_stream.wait_event(event)
        with torch_npu.npu.stream(stream):
            _ = x + 2
        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertGreater(len(warnings), 0)

    def test_record_stream_partial_coverage(self):
        """record_stream for one stream should not cover another stream."""
        x = torch.randn(100, device="npu")
        stream1 = torch_npu.npu.Stream()
        stream2 = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        x.record_stream(stream1)

        stream1.wait_stream(default_stream)
        with torch_npu.npu.stream(stream1):
            _ = x + 1

        stream2.wait_stream(default_stream)
        with torch_npu.npu.stream(stream2):
            _ = x + 2

        warnings = get_event_handler().flush_record_stream_warnings()
        stream1_warnings = [
            w for w in warnings if w.usage_stream == int(stream1.npu_stream)
        ]
        stream2_warnings = [
            w for w in warnings if w.usage_stream == int(stream2.npu_stream)
        ]
        self.assertEqual(len(stream1_warnings), 0)
        self.assertGreater(len(stream2_warnings), 0)


class TestViewAndSlice(SanitizerRecordStreamTestBase):
    def test_view_cross_stream_no_record_stream_warns(self):
        """Cross-stream use of a view should be tracked at storage level."""
        x = torch.randn(100, device="npu")
        view = x[10:50]
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        stream.wait_stream(default_stream)
        with torch_npu.npu.stream(stream):
            _ = view + 1

        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertGreater(len(warnings), 0)

    def test_full_record_stream_covers_view_use(self):
        """record_stream on base tensor should cover view usage."""
        x = torch.randn(100, device="npu")
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        x.record_stream(stream)
        view = x[10:50]

        stream.wait_stream(default_stream)
        with torch_npu.npu.stream(stream):
            _ = view + 1

        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertEqual(len(warnings), 0)

    def test_view_record_stream_covers_full_use(self):
        """record_stream on a view should cover base tensor usage."""
        x = torch.randn(100, device="npu")
        view = x[10:50]
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        view.record_stream(stream)
        stream.wait_stream(default_stream)
        with torch_npu.npu.stream(stream):
            _ = x + 1

        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertEqual(len(warnings), 0)


class TestMemoryReuse(SanitizerRecordStreamTestBase):
    def test_no_stale_recorded_streams_after_realloc(self):
        """A new allocation should not inherit old recorded-stream state."""
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        x = torch.randn(100, device="npu")
        x.record_stream(stream)
        del x

        y = torch.randn(100, device="npu")
        stream.wait_stream(default_stream)

        with torch_npu.npu.stream(stream):
            _ = y + 1

        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertGreater(len(warnings), 0)

    def test_realloc_with_explicit_record_stream(self):
        """A reallocated tensor with explicit record_stream should not warn."""
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        x = torch.randn(100, device="npu")
        del x

        y = torch.randn(100, device="npu")
        y.record_stream(stream)

        stream.wait_stream(default_stream)
        with torch_npu.npu.stream(stream):
            _ = y + 1

        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertEqual(len(warnings), 0)


class TestSanitizerDisabled(TestCase):
    """Behavior when sanitizer is disabled."""
    def setUp(self):
        reset_sanitizer()
        os.environ.pop("TORCH_NPU_SANITIZER", None)

    def test_no_errors_when_disabled(self):
        """Disabled sanitizer should not report cross-stream issues."""
        x = torch.randn(100, device="npu")
        stream = torch_npu.npu.Stream()
        error_raised = False
        try:
            with torch_npu.npu.stream(stream):
                _ = x + 1
            torch_npu.npu.synchronize()
        except Exception:
            error_raised = True

        self.assertFalse(error_raised)


class TestFlushBehavior(SanitizerRecordStreamTestBase):
    def test_dealloc_records_into_error_log(self):
        """Deallocation-time missing-record_stream warnings should be retained."""
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        x = torch.randn(100, device="npu")
        stream.wait_stream(default_stream)
        with torch_npu.npu.stream(stream):
            _ = x + 1
        del x
        gc.collect()
        self.assertGreater(len(get_event_handler().record_stream_errors), 0)


if __name__ == "__main__":
    run_tests()
