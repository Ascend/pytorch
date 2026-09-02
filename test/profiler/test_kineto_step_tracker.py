# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.autograd.profiler import KinetoStepTracker


class TestKinetoStepTracker(TestCase):
    """Unit tests for KinetoStepTracker native behavior contract.

    Verifies that the upstream KinetoStepTracker implementation behaves
    according to its design contract:
    - Global step is monotonically non-decreasing.
    - init_step_count aligns new requester to current global step.
    - increment_step advances per-requester step and updates global max.
    - erase_step_count removes a requester but never rolls back global step.
    - erase_step_count returns bool indicating deletion success.
    """

    def setUp(self):
        # Save snapshot of original global state for post-test restoration.
        self._saved_step_dict = dict(KinetoStepTracker._step_dict)
        self._saved_current_step = KinetoStepTracker._current_step

        # Reset to clean initial state to ensure deterministic test results.
        KinetoStepTracker._step_dict.clear()
        KinetoStepTracker._current_step = 0

    def tearDown(self):
        # Fully restore original global state to prevent cross-test pollution.
        KinetoStepTracker._step_dict.clear()
        KinetoStepTracker._step_dict.update(self._saved_step_dict)
        KinetoStepTracker._current_step = self._saved_current_step

    def test_init_does_not_alter_global_step(self):
        """init_step_count only registers requester, never changes global step."""
        self.assertEqual(KinetoStepTracker.current_step(), 0)

        KinetoStepTracker.init_step_count("req1")
        self.assertEqual(KinetoStepTracker.current_step(), 0)

        # Global step advances only through increment_step
        KinetoStepTracker.increment_step("req1")
        self.assertEqual(KinetoStepTracker.current_step(), 1)

        # New requester aligns to current step; global max remains unchanged
        KinetoStepTracker.init_step_count("req2")
        self.assertEqual(KinetoStepTracker.current_step(), 1)
        self.assertEqual(KinetoStepTracker._step_dict.get("req2"), 1)

    def test_increment_takes_maximum(self):
        """current_step always equals the maximum step among all requesters."""
        KinetoStepTracker.init_step_count("req1")
        KinetoStepTracker.init_step_count("req2")
        self.assertEqual(KinetoStepTracker.current_step(), 0)

        KinetoStepTracker.increment_step("req1")
        self.assertEqual(KinetoStepTracker.current_step(), 1)

        KinetoStepTracker.increment_step("req2")
        KinetoStepTracker.increment_step("req2")
        self.assertEqual(KinetoStepTracker.current_step(), 2)

    def test_erase_keeps_step_monotonic(self):
        """Erasing any requester never decreases global step (monotonic contract)."""
        KinetoStepTracker.init_step_count("req1")
        KinetoStepTracker.init_step_count("req2")

        KinetoStepTracker.increment_step("req1")
        KinetoStepTracker.increment_step("req1")  # req1=2, req2=0
        self.assertEqual(KinetoStepTracker.current_step(), 2)

        # Erasing the max-step requester does not roll back global step
        KinetoStepTracker.erase_step_count("req1")
        self.assertEqual(KinetoStepTracker.current_step(), 2)

    def test_erase_all_requesters_keeps_history(self):
        """Global step retains historical maximum even after all requesters are erased."""
        KinetoStepTracker.init_step_count("req1")
        for _ in range(3):
            KinetoStepTracker.increment_step("req1")
        self.assertEqual(KinetoStepTracker.current_step(), 3)

        KinetoStepTracker.erase_step_count("req1")
        # Step does not roll back even with no active requester
        self.assertEqual(KinetoStepTracker.current_step(), 3)

    def test_erase_return_value_contract(self):
        """erase_step_count returns bool indicating whether deletion succeeded."""
        KinetoStepTracker.init_step_count("req1")

        # Returns True for an existing requester
        self.assertTrue(KinetoStepTracker.erase_step_count("req1"))
        # Returns False for an already-deleted requester
        self.assertFalse(KinetoStepTracker.erase_step_count("req1"))
        # Returns False for a non-existent requester
        self.assertFalse(KinetoStepTracker.erase_step_count("nonexistent"))

    def test_reinit_is_idempotent(self):
        """Calling init_step_count repeatedly on the same requester has no side effect."""
        KinetoStepTracker.init_step_count("req1")
        KinetoStepTracker.increment_step("req1")
        self.assertEqual(KinetoStepTracker.current_step(), 1)
        self.assertEqual(KinetoStepTracker._step_dict.get("req1"), 1)

        # Re-initializing an existing requester does not change its step count
        KinetoStepTracker.init_step_count("req1")
        self.assertEqual(KinetoStepTracker.current_step(), 1)
        self.assertEqual(KinetoStepTracker._step_dict.get("req1"), 1)

    def test_new_requester_inherits_current_step(self):
        """New requester inherits current global step for alignment."""
        KinetoStepTracker.init_step_count("req1")
        for _ in range(5):
            KinetoStepTracker.increment_step("req1")
        self.assertEqual(KinetoStepTracker.current_step(), 5)

        KinetoStepTracker.erase_step_count("req1")
        KinetoStepTracker.init_step_count("req2")
        # Global step stays at historical maximum
        self.assertEqual(KinetoStepTracker.current_step(), 5)
        # New requester aligns to the current global step
        self.assertEqual(KinetoStepTracker._step_dict.get("req2"), 5)


if __name__ == "__main__":
    run_tests()
