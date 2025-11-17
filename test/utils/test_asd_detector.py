import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.utils._asd_detector import set_asd_loss_scale, register_asd_hook


class AsdDetector(TestCase):

    def test_register_asd_hook_with_conditions(self):
        original_func = torch_npu._C._get_silent_check_version

        def mock_get_silent_check_version():
            return 1

        torch_npu._C._get_silent_check_version = mock_get_silent_check_version

        try:
            x = torch.tensor([1.0, 2.0], requires_grad=True)
            weight = torch.tensor([1.0])
            self.assertIsNone(x._backward_hooks)
            result = register_asd_hook(x, weight)

            self.assertIsNone(result)
            self.assertIsNotNone(x._backward_hooks)
        finally:
            torch_npu._C._get_silent_check_version = original_func

    def test_register_asd_hook_early_return(self):
        original_func = torch_npu._C._get_silent_check_version

        def mock_get_silent_check_version():
            return 2

        torch_npu._C._get_silent_check_version = mock_get_silent_check_version

        try:
            x = torch.tensor([1.0, 2.0])
            weight = torch.tensor([1.0])
            result = register_asd_hook(x, weight)
            self.assertIsNone(result)
        finally:
            torch_npu._C._get_silent_check_version = original_func

    def test_set_asd_loss_scale_early_return(self):
        original_func = torch_npu._C._get_silent_check_version

        def mock_get_silent_check_version():
            return 2

        torch_npu._C._get_silent_check_version = mock_get_silent_check_version

        try:
            result = set_asd_loss_scale(2.0)
            self.assertIsNone(result)
        finally:
            torch_npu._C._get_silent_check_version = original_func


if __name__ == '__main__':
    run_tests()