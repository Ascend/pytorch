import sys
import logging
from unittest import mock
import unittest

import torch
import torch.cuda._sanitizer as csan
from torch.utils._python_dispatch import TorchDispatchMode
import torch_npu
import torch_npu.npu._stream_check as stream_check
from torch_npu.testing.testcase import TestCase, run_tests


class TestStreamCheck(TestCase):
    @unittest.skip("disabled now")
    def test_parse_methods_with_valid_inputs(self):
        mock_event_handler = mock.MagicMock()
        mode = stream_check.NPUSanitizerDispatchMode(mock_event_handler)

        mock_schema = mock.MagicMock()
        args = (torch.tensor([1.0]), torch.tensor([2.0]))
        kwargs = {'test_args': torch.tensor([3.0])}

        mode.args_handler = mock.MagicMock()
        mode.parse_inputs(mock_schema, args, kwargs)
        mode.args_handler.parse_inputs.assert_called_once_with(mock_schema, args, kwargs)
        mock_outputs = [torch.tensor([4.0])]
        mode.parse_outputs(mock_outputs)
        mode.args_handler.parse_outputs.assert_called_once_with(mock_outputs)

    @unittest.skip("disabled now")
    def test_torch_dispatch_success(self):
        mock_event_handler = mock.MagicMock()
        mode = stream_check.NPUSanitizerDispatchMode(mock_event_handler)
        mock_func = mock.MagicMock()
        mock_func.__name__ = "aten::add"
        mock_func._schema = mock.MagicMock()
        mock_args = (torch.tensor([1.0]), torch.tensor([2.0]))
        mock_kwargs = {}

        with mock.patch('torch_npu.npu.current_stream') as mock_stream:
            mock_stream_instance = mock.MagicMock()
            mock_stream_instance.npu_stream = 1
            mock_stream.return_value = mock_stream_instance

            with mock.patch.object(mode, 'parse_inputs') as mock_parse_inputs, \
            mock.patch.object(mode, 'parse_outputs') as mock_parse_outputs, \
            mock.patch.object(mode, 'check_errors') as mock_check_errors:
                result = mode.__torch_dispatch__(mock_func, [], mock_args, mock_kwargs)
                mock_parse_inputs.assert_called_once()
                mock_parse_outputs.assert_called_once()
                mock_check_errors.assert_called_once()

    def test_enable_autograd_with_matching_api(self):
        mock_event_handler = mock.MagicMock()
        mode = stream_check.NPUSanitizerDispatchMode(mock_event_handler)
        with mock.patch('torch._C._dispatch_tls_set_dispatch_key_excluded') as mock_set_dispatch:
            mode.enable_autograd("adaptive_avg_pool2d")
            mock_set_dispatch.assert_called_once_with(torch._C.DispatchKey.AutogradFunctionality, False)

    def test_init_with_event_handler(self):
        mock_event_handler = mock.MagicMock()
        mode = stream_check.NPUSanitizerDispatchMode(mock_event_handler)
        self.assertEqual(mode.event_handler, mock_event_handler)
        self.assertIsNone(mode.args_handler)
        self.assertEqual(mode.npu_adjust_autograd, ["adaptive_avg_pool2d", "batch_norm", "log_softmax", "nll_loss", "to"])


if __name__ == "__main__":
    run_tests()