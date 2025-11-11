import os
import atexit
from unittest.mock import MagicMock
from unittest.mock import patch

import torch.cuda._sanitizer as csan
import torch_npu
import torch_npu.utils._npu_trace as npu_trace
import torch_npu.npu._stream_check as stream_check
import torch_npu.npu._kernel_check as kernel_check
from torch_npu.utils.utils import _print_warn_log
import torch_npu.npu._sanitizer as sanitizer
from torch_npu.testing.testcase import TestCase, run_tests


class TestSanitizer(TestCase):
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
                 patch('torch_npu.utils._npu_trace.register_callback_for_npu_event_synchronization'):
                mock_dispatch_instance = mock_dispach.return_value
                mock_dispatch_instance.__enter__.return_value = None
                sanitizer.npu_sanitizer.enable()
                self.assertTrue(sanitizer.npu_sanitizer.enable)
                self.assertEqual(sanitizer.npu_sanitizer.mode, sanitizer.SanitizerMode.STREAM)


if __name__ == "__main__":
    run_tests()