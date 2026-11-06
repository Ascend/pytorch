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

"""
Add validation cases for torch.utils.benchmark APIs on NPU:
1. PyTorch community lacks direct test cases for
   torch.utils.benchmark.Compare.extend_results,
   torch.utils.benchmark.Compare.highlight_warnings,
   torch.utils.benchmark.Compare.print, torch.utils.benchmark.Measurement.merge
   and torch.utils.benchmark.Measurement.significant_figures, so this file is added.
2. This file validates torch.utils.benchmark.Compare,
   torch.utils.benchmark.Measurement (extendable).
"""

import io
from contextlib import redirect_stdout

import torch
import torch.utils.benchmark as benchmark_utils
from torch.testing._internal.common_utils import TestCase, run_tests

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestBenchmarkCompare(TestCase):
    """Test torch.utils.benchmark.Compare methods."""

    def _make_measurement(self, sub_label, raw_times, number_per_run=1):
        task_spec = benchmark_utils.TaskSpec(
            stmt="x + y",
            setup="pass",
            label="add",
            sub_label=sub_label,
            description="(1, 1)",
        )
        return benchmark_utils.Measurement(
            number_per_run=number_per_run,
            raw_times=raw_times,
            task_spec=task_spec,
        )

    def test_extend_results_appends_and_renders(self):
        """Verify extend_results appends NPU measurements and renders them."""
        x = torch.ones(64, 64, device=device_type)
        first = benchmark_utils.Timer(
            "x + y", globals={"x": x, "y": x}, label="add",
            sub_label="case_a", description="(64, 64)")
        second = benchmark_utils.Timer(
            "x + y", globals={"x": x, "y": x}, label="add",
            sub_label="case_b", description="(64, 64)")

        compare = benchmark_utils.Compare([first.timeit(5)])
        compare.extend_results([])
        self.assertNotIn("case_b", str(compare))

        compare.extend_results([second.timeit(5)])
        table = str(compare)
        self.assertIn("case_a", table)
        self.assertIn("case_b", table)

    def test_extend_results_rejects_non_measurement(self):
        """Verify extend_results raises ValueError for non Measurement input."""
        compare = benchmark_utils.Compare([])

        with self.assertRaises(ValueError):
            compare.extend_results([1])

    def test_highlight_warnings_adds_footer_for_high_variance(self):
        """Verify highlight_warnings adds a footer for high variance measurements."""
        stable = self._make_measurement("stable", [1.0e-3] * 7)
        noisy = self._make_measurement("noisy", [1.0e-3] * 4 + [1.0e-1] * 4)
        compare = benchmark_utils.Compare([stable, noisy])

        self.assertNotIn("high variance", str(compare))

        compare.highlight_warnings()
        self.assertIn("high variance", str(compare))

    def test_highlight_warnings_no_footer_for_stable_measurements(self):
        """Verify highlight_warnings adds no footer when all measurements are stable."""
        stable = self._make_measurement("stable", [1.0e-3] * 7)
        compare = benchmark_utils.Compare([stable])
        compare.highlight_warnings()

        self.assertNotIn("high variance", str(compare))

    def test_print_outputs_formatted_table(self):
        """Verify print outputs the same content as the string representation."""
        compare = benchmark_utils.Compare(
            [self._make_measurement("stable", [1.0e-3] * 7)])

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            compare.print()

        self.assertEqual(buffer.getvalue().strip(), str(compare).strip())


class TestBenchmarkMeasurement(TestCase):
    """Test torch.utils.benchmark.Measurement methods."""

    def _make_measurement(self, sub_label, raw_times, number_per_run=1, metadata=None):
        task_spec = benchmark_utils.TaskSpec(
            stmt="x + y",
            setup="pass",
            label="add",
            sub_label=sub_label,
            description="(1, 1)",
        )
        return benchmark_utils.Measurement(
            number_per_run=number_per_run,
            raw_times=raw_times,
            task_spec=task_spec,
            metadata=metadata,
        )

    def test_merge_groups_measurements_with_same_task_spec(self):
        """Verify merge groups replicates sharing one task spec into one measurement."""
        m1 = self._make_measurement(
            "replicate", [10e-3, 20e-3], number_per_run=10, metadata={"k": 1})
        m2 = self._make_measurement("replicate", [4e-3, 8e-3], number_per_run=2)

        merged = benchmark_utils.Measurement.merge((m1, m2))

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].number_per_run, 1)
        self.assertEqual(merged[0].raw_times, m1.times + m2.times)
        self.assertIsNone(merged[0].metadata)

    def test_merge_keeps_measurements_with_distinct_task_specs(self):
        """Verify merge keeps measurements with distinct task specs separate."""
        m1 = self._make_measurement("first", [1.0e-3])
        m2 = self._make_measurement("second", [2.0e-3])

        merged = benchmark_utils.Measurement.merge((m1, m2))

        self.assertEqual(len(merged), 2)
        self.assertEqual({m.task_spec.sub_label for m in merged}, {"first", "second"})

    def test_merge_npu_timer_measurements(self):
        """Verify merge combines multiple NPU timer runs into one measurement."""
        x = torch.ones(128, 128, device=device_type)
        timer = benchmark_utils.Timer(
            "x + y", globals={"x": x, "y": x}, label="add", description="(128, 128)")
        runs = [timer.timeit(5) for _ in range(3)]

        merged = benchmark_utils.Measurement.merge(runs)

        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0].raw_times), 3)
        self.assertGreater(merged[0].median, 0)

    def test_significant_figures_range_and_ordering(self):
        """Verify significant_figures is within [1, 4] and reflects spread."""
        tight = self._make_measurement("tight", [1.0] * 24)
        loose = self._make_measurement("loose", [1.0] * 12 + [100.0] * 12)

        for measurement in (tight, loose):
            self.assertIsInstance(measurement.significant_figures, int)
            self.assertGreaterEqual(measurement.significant_figures, 1)
            self.assertLessEqual(measurement.significant_figures, 4)

        self.assertGreater(tight.significant_figures, loose.significant_figures)

    def test_significant_figures_npu_measurement(self):
        """Verify significant_figures works for a merged NPU timer measurement."""
        x = torch.ones(64, 64, device=device_type)
        timer = benchmark_utils.Timer(
            "x + y", globals={"x": x, "y": x}, label="add", description="(64, 64)")
        merged = benchmark_utils.Measurement.merge([timer.timeit(5) for _ in range(4)])

        sigfig = merged[0].significant_figures

        self.assertIsInstance(sigfig, int)
        self.assertGreaterEqual(sigfig, 1)
        self.assertLessEqual(sigfig, 4)


if __name__ == "__main__":
    run_tests()
