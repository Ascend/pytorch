import os
from unittest.mock import MagicMock
from unittest.mock import patch

import torch
import torch.utils.cpp_extension

import torch_npu
import torch_npu.npu._sanitizer as sanitizer
from torch_npu.testing.testcase import TestCase, run_tests


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTORCH_INSTALL_PATH = os.path.dirname(os.path.realpath(torch.__file__))
PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_npu.__file__))

   
class TestSanitizer(TestCase):
    def tearDown(self):
        if sanitizer.npu_sanitizer.dispatch is not None:
            try:
                sanitizer.npu_sanitizer.dispatch.__exit__(None, None, None)
            except Exception:
                pass
        sanitizer.npu_sanitizer.dispatch = None
        sanitizer.npu_sanitizer.event_handler = None
        sanitizer.npu_sanitizer.enabled = False

    def test_del_method_with_dispatch(self):
        mock_dispatch = MagicMock()
        sanitizer.npu_sanitizer.dispatch = mock_dispatch
        sanitizer.npu_sanitizer.__del__()
        mock_dispatch.__exit__.assert_called_once_with(None, None, None)

    def test_enable_kernel_check_no_debug_path(self):
        with patch.dict(os.environ, {}, clear=True):
            result = sanitizer.npu_sanitizer.enable_kernel_check()
            self.assertFalse(result)

    def test_enable_stream_check_mode(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch('torch.cuda._sanitizer.EventHandler') as mock_event_handler, \
                 patch('torch_npu.npu._stream_check.NPUSanitizerDispatchMode') as mock_dispach, \
                 patch('torch_npu.utils._npu_trace.register_callback_for_npu_event_creation'), \
                 patch('torch_npu.utils._npu_trace.register_callback_for_npu_event_deletion'), \
                 patch('torch_npu.utils._npu_trace.register_callback_for_npu_event_record'), \
                 patch('torch_npu.utils._npu_trace.register_callback_for_npu_event_wait'), \
                 patch('torch_npu.utils._npu_trace.register_callback_for_npu_memory_allocation'), \
                 patch('torch_npu.utils._npu_trace.register_callback_for_npu_memory_deallocation'), \
                 patch('torch_npu.utils._npu_trace.register_callback_for_npu_stream_creation'), \
                 patch('torch_npu.utils._npu_trace.register_callback_for_npu_device_synchronization'), \
                 patch('torch_npu.utils._npu_trace.register_callback_for_npu_stream_synchronization'), \
                 patch('torch_npu.utils._npu_trace.register_callback_for_npu_event_synchronization'),  \
                 patch("torch_npu.utils._npu_trace.register_callback_for_npu_record_stream"), \
                 patch("torch_npu.utils._npu_trace.register_callback_for_npu_erase_stream"):
                mock_dispatch_instance = mock_dispach.return_value
                mock_dispatch_instance.__enter__.return_value = None
                sanitizer.npu_sanitizer.enable()
                self.assertTrue(sanitizer.npu_sanitizer.enable)
                self.assertEqual(sanitizer.npu_sanitizer.mode, sanitizer.SanitizerMode.STREAM)


class TestSanitizerRegressionGuards(TestCase):
    """Regression guards for sanitizer integration issues."""

    def test_pluggable_allocator_traces_alloc_and_free(self):
        """Custom allocator must emit alloc/free traces, not only record_stream traces."""
        source_path = os.path.join(
            REPO_ROOT, "torch_npu", "csrc", "npu", "NPUPluggableAllocator.cpp"
        )
        with open(source_path, encoding="utf-8") as source_file:
            source = source_file.read()

        self.assertIn(
            "traceNpuMemoryAllocation",
            source,
            "NPUPluggableAllocator allocations must be visible to NPU Sanitizer",
        )
        self.assertIn(
            "traceNpuMemoryDeallocation",
            source,
            "NPUPluggableAllocator frees must trigger deferred record_stream checks",
        )

    def test_caching_allocator_traces_erase_stream_paths(self):
        """NPUCachingAllocator eraseStream paths should be visible to NPU Sanitizer."""
        source_path = os.path.join(
            REPO_ROOT, "torch_npu", "csrc", "core", "npu", "NPUCachingAllocator.cpp"
        )
        with open(source_path, encoding="utf-8") as source_file:
            source = source_file.read()

        self.assertIn(
            "traceNpuEraseStream",
            source,
            "NPUCachingAllocator eraseStream / eraseStreamWithBlockPtr should be visible "
            "to NPU Sanitizer so recorded_streams can be cleared.",
        )
        self.assertIn(
            "eraseStreamWithBlockPtr",
            source,
            "MULTI_STREAM_MEMORY_REUSE=3 optimized block-ptr erase path should remain covered.",
        )


if __name__ == "__main__":
    run_tests()