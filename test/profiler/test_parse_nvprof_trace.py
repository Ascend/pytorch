# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Owner(s): ["oncall: profiler"]
"""Ascend NPU adaptation checks for torch.autograd.profiler.parse_nvprof_trace.

``torch.autograd.profiler.parse_nvprof_trace`` / ``load_nvprof`` parse NVIDIA
nvprof CUPTI SQLite. Ascend does not emit that format; NPU profiling uses
``torch.autograd.profiler.profile(use_device="npu")``.

Community pytorch has limited CUPTI-oriented coverage and does not provide an
Ascend/NPU adaptation suite for this API. This file therefore:
  1. checks the APIs remain available under the torch_npu stack;
  2. verifies NPU profiler semantics with real NPU ops;
  3. exercises ``EnforceUnique`` (used inside ``parse_nvprof_trace``);
  4. confirms ``parse_nvprof_trace`` / ``load_nvprof`` still reject missing
     nvprof DB paths after NPU profiling (negative path; no CUPTI fixture).
"""

import os
import tempfile

import torch
from torch.autograd.profiler import EnforceUnique, load_nvprof, parse_nvprof_trace
from torch.testing._internal.common_utils import TestCase, run_tests


class TestParseNvprofTraceAscendNPU(TestCase):
    """NPU checks for parse_nvprof_trace availability and NPU profiler semantics."""

    def setUp(self):
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            self.skipTest("Requires Ascend NPU (CANN). Skip on CPU/CUDA-only hosts.")
        torch.npu.set_device(0)
        self.device = torch.device("npu:0")

    def _mm_on_npu(self):
        a = torch.randn(8, 8, device=self.device)
        b = torch.randn(8, 8, device=self.device)
        return torch.mm(a, b)

    def _event_names(self, prof):
        return [e.name for e in prof.function_events]

    def test_api_available_on_npu_stack(self):
        self.assertTrue(hasattr(torch.autograd.profiler, "parse_nvprof_trace"))
        self.assertTrue(callable(torch.autograd.profiler.parse_nvprof_trace))
        self.assertTrue(hasattr(torch.autograd.profiler, "load_nvprof"))
        self.assertTrue(callable(torch.autograd.profiler.load_nvprof))
        self.assertIs(parse_nvprof_trace, torch.autograd.profiler.parse_nvprof_trace)
        self.assertIs(load_nvprof, torch.autograd.profiler.load_nvprof)

    def test_npu_profiler_records_mm(self):
        """Real NPU profiler path: use_device='npu' must capture mm events."""
        with torch.autograd.profiler.profile(use_device="npu") as prof:
            out = self._mm_on_npu()
            torch.npu.synchronize()
        self.assertEqual(out.device.type, "npu")
        self.assertTrue(
            any("mm" in name for name in self._event_names(prof)),
            f"expected mm in NPU profiler events, got {self._event_names(prof)}",
        )

    def test_npu_profiler_records_add_and_mm(self):
        """NPU profiler should record multiple ops executed on device."""
        with torch.autograd.profiler.profile(use_device="npu") as prof:
            a = torch.randn(16, 16, device=self.device)
            b = torch.randn(16, 16, device=self.device)
            c = torch.mm(a, b)
            d = c + a
            torch.npu.synchronize()
        self.assertEqual(d.device.type, "npu")
        names = self._event_names(prof)
        self.assertTrue(any("mm" in n for n in names), f"missing mm in {names}")
        self.assertTrue(
            any(("add" in n) or ("+" in n) for n in names),
            f"missing add in {names}",
        )

    def test_npu_profiler_event_timing(self):
        """Captured NPU profiler events should expose non-negative CPU time."""
        with torch.autograd.profiler.profile(use_device="npu") as prof:
            _ = self._mm_on_npu()
            torch.npu.synchronize()
        mm_events = [e for e in prof.function_events if "mm" in e.name]
        self.assertGreater(len(mm_events), 0)
        for evt in mm_events:
            self.assertGreaterEqual(evt.cpu_time, 0.0)

    def test_parse_apis_reject_missing_nvprof_db_after_npu_profile(self):
        """After NPU profiling, parse APIs remain callable and reject missing DB."""
        with torch.autograd.profiler.profile(use_device="npu") as prof:
            _ = self._mm_on_npu()
            torch.npu.synchronize()
        self.assertGreater(len(list(prof.function_events)), 0)

        missing = os.path.join(tempfile.gettempdir(), "missing_nvprof_ascend.sqlite")
        with self.assertRaises(Exception):
            parse_nvprof_trace(missing)
        with self.assertRaises(Exception):
            load_nvprof(missing)

    def test_enforce_unique_on_npu(self):
        _ = torch.randn(2, 2, device=self.device)
        unique = EnforceUnique()
        unique.see("a", 1)
        with self.assertRaises(RuntimeError):
            unique.see("a", 1)


if __name__ == "__main__":
    run_tests()
