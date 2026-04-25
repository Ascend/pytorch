"""
Add validation cases for torch.hub APIs on NPU:
1. test/test_hub.py from PyTorch community lacks sufficient API validations for torch.hub.help and torch.hub._get_torch_home, so this file is added.
2. This file validates torch.hub.help, torch.hub._get_torch_home (extendable).
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch.hub as hub
from torch.testing._internal.common_utils import TestCase, run_tests


class TestHubHelp(TestCase):
    """Test torch.hub.help API with zero external dependency."""

    def setUp(self):
        self.mock_repo_obj = tempfile.TemporaryDirectory()
        self.mock_repo = self.mock_repo_obj.name
        self.addCleanup(self.mock_repo_obj.cleanup)

        # Clean up sys.path pollution after each test
        self.addCleanup(lambda: sys.path.remove(self.mock_repo) if self.mock_repo in sys.path else None)

        hubconf_path = os.path.join(self.mock_repo, "hubconf.py")
        with open(hubconf_path, "w", encoding="utf-8") as f:
            f.write(
                'def entry_with_docstring():\n'
                '    """This is a mock docstring containing EfficientNet info."""\n'
                '    pass\n\n'
                'def entry_without_docstring():\n'
                '    pass\n'
            )
        os.makedirs(os.path.join(self.mock_repo, ".git"), exist_ok=True)

    def test_help_function_callable(self):
        """Verify help function exists and is callable."""
        self.assertTrue(hasattr(hub, "help"))
        self.assertTrue(callable(hub.help))

    @patch("torch.hub._get_cache_or_reload")
    def test_help_returns_none_without_docstring(self, mock_get_repo):
        """Verify help returns None when entrypoint has no docstring."""
        mock_get_repo.return_value = self.mock_repo
        docstring = hub.help(
            "mock/local_repo",
            "entry_without_docstring",
            force_reload=False,
            trust_repo=True,
        )
        self.assertIsNone(docstring)

    @patch("torch.hub._get_cache_or_reload")
    def test_help_returns_docstring_with_content(self, mock_get_repo):
        """Verify help returns valid docstring when entrypoint has docstring."""
        mock_get_repo.return_value = self.mock_repo
        docstring = hub.help(
            "mock/local_repo",
            "entry_with_docstring",
            force_reload=False,
            trust_repo=True,
        )
        self.assertIsInstance(docstring, str)
        self.assertTrue(len(docstring) > 0)
        self.assertIn("EfficientNet", docstring)


class TestHubGetTorchHome(TestCase):
    """Test torch.hub._get_torch_home API"""

    def test_get_torch_home_returns_path(self):
        """Verify _get_torch_home returns a valid path string."""
        torch_home = hub._get_torch_home()
        self.assertIsInstance(torch_home, str)
        self.assertTrue(len(torch_home) > 0)

    def test_get_torch_home_with_env_variable(self):
        """Verify _get_torch_home respects TORCH_HOME environment variable."""
        original = os.environ.get("TORCH_HOME")
        self.addCleanup(
            lambda: os.environ.pop("TORCH_HOME", None) if original is None
            else os.environ.__setitem__("TORCH_HOME", original)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["TORCH_HOME"] = tmpdir
            if hasattr(hub._get_torch_home, "cache_clear"):
                hub._get_torch_home.cache_clear()
            self.assertEqual(hub._get_torch_home(), tmpdir)


if __name__ == "__main__":
    run_tests()