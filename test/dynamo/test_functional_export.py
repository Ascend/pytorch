"""
Add focused validation cases for TorchDynamo APIs.

1. These APIs are exercised indirectly by existing PyTorch tests, so this file
   adds dedicated API-level validation.
2. This file validates the following APIs:
   torch._dynamo.functional_export.dynamo_graph_capture_for_export
   torch._dynamo.utils.is_compile_supported
"""

from unittest.mock import patch

import torch
from torch._dynamo.functional_export import dynamo_graph_capture_for_export
from torch._dynamo.utils import is_compile_supported
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
)


class TestDynamoGraphCaptureForExport(TestCase):
    """Test torch._dynamo.functional_export.dynamo_graph_capture_for_export."""

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def test_capture_simple_function(self):
        def fn(x, y):
            return torch.relu(x + y * 2)

        x = torch.randn(2, 3).to(device_type)
        y = torch.randn(2, 3).to(device_type)
        expected = fn(x, y)

        graph_module = dynamo_graph_capture_for_export(fn)(x, y)
        actual = graph_module(x, y)

        self.assertIsInstance(graph_module, torch.fx.GraphModule)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.device.type, device_type)

    def test_capture_module_with_parameter_and_buffer(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(3, 3))
                self.register_buffer("bias", torch.randn(3))

            def forward(self, x):
                return x @ self.weight + self.bias

        module = TestModule().to(device_type).eval()
        x = torch.randn(4, 3).to(device_type)
        expected = module(x)

        graph_module = dynamo_graph_capture_for_export(module)(x)
        actual = graph_module(x)

        self.assertIsInstance(graph_module, torch.fx.GraphModule)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.device.type, device_type)


class TestIsCompileSupported(TestCase):
    """Test torch._dynamo.utils.is_compile_supported."""

    @patch("torch._dynamo.eval_frame.is_dynamo_supported", return_value=True)
    def test_cpu_supported_when_dynamo_is_supported(
        self, mock_is_dynamo_supported
    ):
        self.assertTrue(is_compile_supported("cpu"))
        mock_is_dynamo_supported.assert_called_once_with()

    @patch("torch._dynamo.eval_frame.is_dynamo_supported", return_value=False)
    def test_cpu_unsupported_when_dynamo_is_unsupported(
        self, mock_is_dynamo_supported
    ):
        self.assertFalse(is_compile_supported("cpu"))
        mock_is_dynamo_supported.assert_called_once_with()

    @patch("torch._dynamo.utils.has_triton", return_value=True)
    @patch("torch._dynamo.eval_frame.is_dynamo_supported", return_value=True)
    def test_cuda_supported_with_triton(
        self, mock_is_dynamo_supported, mock_has_triton
    ):
        self.assertTrue(is_compile_supported("cuda"))
        mock_is_dynamo_supported.assert_called_once_with()
        mock_has_triton.assert_called_once_with()

    @patch("torch._dynamo.utils.has_triton", return_value=False)
    @patch("torch._dynamo.eval_frame.is_dynamo_supported", return_value=True)
    def test_cuda_unsupported_without_triton(
        self, mock_is_dynamo_supported, mock_has_triton
    ):
        self.assertFalse(is_compile_supported("cuda"))
        mock_is_dynamo_supported.assert_called_once_with()
        mock_has_triton.assert_called_once_with()

    @patch("torch._dynamo.eval_frame.is_dynamo_supported", return_value=True)
    def test_npu_returns_false(self, mock_is_dynamo_supported):
        self.assertFalse(is_compile_supported("npu"))
        mock_is_dynamo_supported.assert_called_once_with()


if __name__ == "__main__":
    run_tests()
