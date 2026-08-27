"""
Add focused validation cases for TorchDynamo APIs.

1. This API is exercised indirectly by existing PyTorch tests, so this file
   adds dedicated API-level validation.
2. This file validates torch._dynamo.utils.is_compile_supported.
"""

from unittest.mock import patch

from torch._dynamo.utils import is_compile_supported
from torch.testing._internal.common_utils import TestCase, run_tests


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