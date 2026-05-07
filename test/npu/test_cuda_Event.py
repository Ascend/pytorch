# Owner(s): ["module: tests"]

import torch_npu  # noqa: F401

import torch


try:
    from torch_npu.testing.testcase import run_tests, TestCase
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestCudaEvent(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_npu_create_event(self):
        """Verify Event instance can be created with default args."""
        e = torch.npu.Event()
        self.assertIsInstance(e, torch.npu.Event)

    def test_npu_event_query_returns_bool(self):
        """Verify query() returns a boolean value."""
        e = torch.npu.Event()
        result = e.query()
        self.assertIsInstance(result, bool)

    def test_npu_event_wait(self):
        """Verify wait() completes without error."""
        e = torch.npu.Event()
        e.wait()

    def test_npu_event_record(self):
        """Verify record() on a stream completes without error."""
        e = torch.npu.Event()
        s = torch.npu.default_stream()
        e.record(s)

    def test_npu_event_enable_timing(self):
        """Verify Event with enable_timing=True can be created."""
        e = torch.npu.Event(enable_timing=True)
        self.assertIsInstance(e, torch.npu.Event)

    def test_npu_event_blocking(self):
        """Verify Event with blocking=True can be created."""
        e = torch.npu.Event(blocking=True)
        self.assertIsInstance(e, torch.npu.Event)

    def test_npu_event_query_after_record(self):
        """Verify query() returns True after record + synchronize."""
        e = torch.npu.Event()
        s = torch.npu.default_stream()
        e.record(s)
        s.synchronize()
        result = e.query()
        # Event should be marked as completed after stream synchronization
        self.assertTrue(result)


if __name__ == "__main__":
    run_tests()
