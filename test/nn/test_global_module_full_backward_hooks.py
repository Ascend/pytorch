"""
Add validation cases for torch.nn global module hook APIs on NPU:

1. PyTorch community lacks direct validations for some global backward hook APIs.
2. This file validates torch.nn.modules.module.register_module_full_backward_hook and
   torch.nn.modules.module.register_module_full_backward_pre_hook.

"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestGlobalModuleFullBackwardHooks(TestCase):

    def test_register_module_full_backward_hook(self):
        module = torch.nn.Sigmoid().to(device_type)
        inp = torch.randn(5, 5, device=device_type, requires_grad=True)
        sig_x = torch.sigmoid(inp)
        calls = []

        def hook(mod, grad_input, grad_output):
            if isinstance(mod, torch.nn.Sigmoid):
                calls.append(mod)
                return (grad_input[0] * 2,)
            return None

        handle = torch.nn.modules.module.register_module_full_backward_hook(hook)
        try:
            module(inp).backward(torch.ones(5, 5, device=device_type))
        finally:
            handle.remove()

        self.assertEqual(len(calls), 1)
        self.assertEqual(inp.grad, sig_x * (1 - sig_x) * 2)

    def test_register_module_full_backward_pre_hook(self):
        module = torch.nn.Sigmoid().to(device_type)
        inp = torch.randn(5, 5, device=device_type, requires_grad=True)
        sig_x = torch.sigmoid(inp)
        calls = []

        def hook(mod, grad_output):
            if isinstance(mod, torch.nn.Sigmoid):
                calls.append(mod)
                return (grad_output[0] * 0.5,)
            return None

        handle = torch.nn.modules.module.register_module_full_backward_pre_hook(hook)
        try:
            module(inp).backward(torch.ones(5, 5, device=device_type))
        finally:
            handle.remove()

        self.assertEqual(len(calls), 1)
        self.assertEqual(inp.grad, sig_x * (1 - sig_x) * 0.5)


if __name__ == "__main__":
    run_tests()
