# Owner(s): ["oncall: jit"]
"""
Add validation cases for torch.jit APIs:
1. Official jit test files lack sufficient validation for some torch.jit APIs, so this file is added.
2. This file validates:
torch.jit.onednn_fusion_enabled
torch.jit.enable_onednn_fusion
(extendable)
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOneDNNJitAPI(TestCase):
    def setUp(self):
        self.original_state = torch.jit.onednn_fusion_enabled()
        super().setUp()

    def tearDown(self):
        torch.jit.enable_onednn_fusion(self.original_state)
        super().tearDown()

    def test_onednn_fusion_enabled_returns_bool(self):
        result = torch.jit.onednn_fusion_enabled()
        self.assertIsInstance(result, bool)

    def test_onednn_fusion_enable_disable_roundtrip(self):
        torch.jit.enable_onednn_fusion(True)
        self.assertEqual(torch.jit.onednn_fusion_enabled(), True)

        torch.jit.enable_onednn_fusion(False)
        self.assertEqual(torch.jit.onednn_fusion_enabled(), False)


if __name__ == "__main__":
    run_tests()
