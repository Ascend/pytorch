import os
import sys
import torch

from torch_npu.utils._path_manager import PathManager
from torch_npu.profiler._dynamic_profiler._dynamic_profiler_utils import DynamicProfilerUtils
from torch_npu.profiler.dynamic_profile import init as dp_init
from torch_npu.profiler.dynamic_profile import step as dp_step
from torch_npu.profiler.analysis.prof_common_func._constant import print_error_msg, print_warn_msg
import torch_npu.profiler._non_intrusive_profile as none_intrusive_profile
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.profiler._non_intrusive_profile import _NonIntrusiveProfile


class TestNoneInstrusiveProfile(TestCase):
    def test_step_wrapper(self):
        optimizer = torch.optim.SGD([torch.tensor([1.0])], lr=0.01)

        def mock_step_func(*args, **kwargs):
            return "wrapped_result"

        wrapped_func = none_intrusive_profile._NonIntrusiveProfile.step_wrapper(mock_step_func)
        result = wrapped_func(optimizer)
        self.assertEqual(result, "wrapped_result")

    def test_check_last_optimizer(self):
        optimizer1 = torch.optim.SGD([torch.tensor([1.0])], lr=0.01)
        optimizer2 = torch.optim.Adam([torch.tensor([1.0])], lr=0.01)

        _NonIntrusiveProfile.OPTIMIZER_ID = id(optimizer1)
        self.assertTrue(_NonIntrusiveProfile.check_last_optimizer(optimizer1))
        self.assertFalse(_NonIntrusiveProfile.check_last_optimizer(optimizer2))

    def test_patch_step_function(self):
        optimizer = torch.optim.SGD([torch.tensor([1.0])], lr=0.01)
        none_intrusive_profile._NonIntrusiveProfile.patch_step_function(optimizer)
        self.assertTrue(hasattr(optimizer.__class__.step, 'step_hooked'))


if __name__ == "__main__":
    run_tests()

