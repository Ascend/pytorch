"""Unit tests for config_optimizer.py — FASTA config pruning pipeline.

Tests the filtering, deduplication, sampling, and expert preservation
logic using static configs in the FastAConfig format (kwargs, from_expert,
circle_num). No NPU hardware or torch.compile required.
"""
import importlib.util
import os
import sys
import types

import torch
from torch.testing._internal.common_utils import (
    run_tests, parametrize, instantiate_parametrized_tests,
)
from testutils import TestUtils

# The optimizer on/off flag (config.fasta_config_optimizer) is provided by the
# stub below, so the FASTA_CONFIG_OPTIMIZER env var is not needed here. The
# pruning bounds (MAX_CIRCLE_NUM, MIN_SUB_NUMEL, MAX_CONFIGS) are private
# constants in config_optimizer and cannot be set by the user.

# config_optimizer imports `from .config import fasta_config_optimizer` and
# `from .fasta_autotune import log`. Importing the real modules pulls in the
# native torch_npu extension. Stub both so we can import config_optimizer by
# path without that dependency.
_cfg_stub = types.ModuleType("torch_npu._inductor.experimental.dynamic_filter.dynamic_filter_config")
_cfg_stub.fasta_config_optimizer = True
sys.modules["torch_npu._inductor.experimental.dynamic_filter.dynamic_filter_config"] = _cfg_stub

_fa_stub = types.ModuleType("torch_npu._inductor.fasta_autotune")
import logging
_fa_stub.log = logging.getLogger("config_optimizer_test")
sys.modules["torch_npu._inductor.fasta_autotune"] = _fa_stub

# Provide the parent package so relative imports resolve.
if "torch_npu._inductor" not in sys.modules:
    _pkg = types.ModuleType("torch_npu._inductor")
    _pkg.__path__ = []
    sys.modules["torch_npu._inductor"] = _pkg

_CO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "torch_npu", "_inductor", "experimental", "dynamic_filter", "config_optimizer.py",
)
_spec = importlib.util.spec_from_file_location(
    "torch_npu._inductor.experimental.dynamic_filter.config_optimizer", os.path.abspath(_CO_PATH))
co = importlib.util.module_from_spec(_spec)
sys.modules["torch_npu._inductor.experimental.dynamic_filter.config_optimizer"] = co
_spec.loader.exec_module(co)


class MockConfig:
    """Static config in the FastAConfig format: kwargs, from_expert, circle_num."""

    def __init__(self, kwargs, from_expert=False, circle_num=-1):
        self.kwargs = kwargs
        self.from_expert = from_expert
        self.circle_num = circle_num

    def __repr__(self):
        return f"MockConfig({self.kwargs}, expert={self.from_expert}, cn={self.circle_num})"


def _make_config(block, sub, from_expert=False, circle_num=-1, axis="X0"):
    """Create a 1-axis config with given BLOCK and BLOCK_SUB values."""
    return MockConfig(
        {f"{axis}BLOCK": block, f"{axis}BLOCK_SUB": sub},
        from_expert=from_expert, circle_num=circle_num,
    )


def _make_2d_config(xblock, xsub, yblock, ysub, from_expert=False):
    """Create a 2-axis config."""
    return MockConfig(
        {"X0BLOCK": xblock, "X0BLOCK_SUB": xsub,
         "Y0BLOCK": yblock, "Y0BLOCK_SUB": ysub},
        from_expert=from_expert,
    )


class TestConfigOptimizer(TestUtils):

    def test_expert_configs_preserved(self):
        """Expert configs must survive all pruning unconditionally."""
        expert = _make_config(128, 64, from_expert=True)
        fasta = _make_config(256, 128, from_expert=False)
        result = co.optimize_configs([expert, fasta])
        experts_out = [c for c in result if c.from_expert]
        self.assertEqual(len(experts_out), 1)
        self.assertIs(experts_out[0], expert)

    def test_expert_with_bad_circle_num_still_preserved(self):
        """Expert configs bypass the circle_num filter."""
        expert = _make_config(128, 8, from_expert=True, circle_num=10)
        result = co.optimize_configs([expert])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], expert)

    @parametrize("circle_num,kept", [(2, True), (4, True), (5, False), (8, False)])
    def test_circle_num_filter(self, circle_num, kept):
        """Native configs are kept iff circle_num <= MAX_CIRCLE_NUM (4)."""
        cfg = _make_config(128, 64, circle_num=circle_num)
        # Anchor expert config so the result is never empty (avoids the safety
        # net that returns the original list when all configs are filtered out).
        anchor = _make_config(256, 128, from_expert=True)
        result = co.optimize_configs([cfg, anchor])
        fasta_out = [c for c in result if not c.from_expert]
        self.assertEqual(cfg in fasta_out, kept)

    def test_circle_num_computed_from_kwargs(self):
        """When circle_num=-1, it is computed from BLOCK/BLOCK_SUB."""
        good = _make_config(256, 64)   # ceil(256/64) = 4 <= 4
        bad = _make_config(256, 16)    # ceil(256/16) = 16 > 4
        result = co.optimize_configs([good, bad])
        fasta_out = [c for c in result if not c.from_expert]
        self.assertIn(good, fasta_out)
        self.assertNotIn(bad, fasta_out)

    def test_min_sub_numel_filter(self):
        """Native configs with sub_numel < MIN_SUB_NUMEL (32) are removed."""
        good = _make_config(128, 64)   # sub_numel=64 >= 32
        bad = _make_config(64, 24)     # cn=ceil(64/24)=3, sub_numel=24 < 32
        result = co.optimize_configs([good, bad])
        fasta_out = [c for c in result if not c.from_expert]
        self.assertIn(good, fasta_out)
        self.assertNotIn(bad, fasta_out)

    def test_min_sub_numel_2d(self):
        """For 2D configs, sub_numel is the product of both BLOCK_SUB values."""
        good = _make_2d_config(64, 32, 64, 32)  # sub=1024, cn=4
        bad = _make_2d_config(16, 4, 16, 4)      # sub=16, cn=16
        result = co.optimize_configs([good, bad])
        fasta_out = [c for c in result if not c.from_expert]
        self.assertIn(good, fasta_out)
        self.assertNotIn(bad, fasta_out)

    def test_dedup(self):
        """Duplicate configs (same circle_num + sub_block pattern) are removed."""
        cfg1 = _make_config(128, 64)
        cfg2 = _make_config(128, 64)
        result = co.optimize_configs([cfg1, cfg2])
        fasta_out = [c for c in result if not c.from_expert]
        self.assertEqual(len(fasta_out), 1)

    def test_sample_diverse_caps(self):
        """When more than MAX_CONFIGS native configs, sampling caps the count."""
        configs = [_make_config(sub, sub) for sub in range(32, 132)]
        result = co.optimize_configs(configs)
        fasta_out = [c for c in result if not c.from_expert]
        self.assertLessEqual(len(fasta_out), 50)
        self.assertGreater(len(fasta_out), 0)

    def test_sample_preserves_diversity(self):
        """Sampling picks from across the sub_numel range, not just one end."""
        configs = [_make_config(sub, sub) for sub in range(32, 1032, 10)]
        result = co.optimize_configs(configs)
        fasta_out = [c for c in result if not c.from_expert]
        sub_numels = sorted(co._get_sub_numel(c) for c in fasta_out)
        self.assertLessEqual(sub_numels[0], 42)
        self.assertGreaterEqual(sub_numels[-1], 1012)

    def test_empty_returns_empty(self):
        """Empty input returns empty output."""
        self.assertEqual(co.optimize_configs([]), [])

    def test_optimizer_disabled(self):
        """When the optimizer flag is off, returns the original list unchanged."""
        original = co.fasta_config_optimizer
        try:
            co.fasta_config_optimizer = False
            configs = [_make_config(128, 64), _make_config(256, 16)]
            result = co.optimize_configs(configs)
            self.assertEqual(len(result), len(configs))
            self.assertIs(result[0], configs[0])
        finally:
            co.fasta_config_optimizer = original

    def test_all_filtered_returns_original(self):
        """If all configs would be filtered, return original as a safety net."""
        configs = [_make_config(256, 8) for _ in range(5)]
        result = co.optimize_configs(configs)
        self.assertEqual(len(result), 5)


instantiate_parametrized_tests(TestConfigOptimizer)

if __name__ == "__main__":
    run_tests()
