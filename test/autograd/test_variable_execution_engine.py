"""
Add validation cases for torch.autograd APIs on NPU:
1. PyTorch community lacks direct API validations for
   torch.autograd.variable.Variable._execution_engine.run_backward, so this file is added.
2. This file validates torch.autograd.variable.Variable._execution_engine.run_backward (extendable).
"""
import torch
from torch.autograd.variable import Variable
from torch.testing._internal.common_utils import TestCase, run_tests

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestVariableExecutionEngineRunBackward(TestCase):

    def test_run_backward_accumulate_grad(self):
        # Same call pattern as torch.autograd.backward, and compare NPU
        # gradients against CPU gradients computed from the same data.
        base = torch.randn(3, 3)
        x_cpu = base.clone().requires_grad_()
        x_npu = base.clone().to(device_type).requires_grad_()
        for x in (x_cpu, x_npu):
            y = (x * x + x).sum()
            Variable._execution_engine.run_backward(
                (y,),
                (torch.ones_like(y),),
                False,  # retain_graph
                False,  # create_graph
                (),     # inputs: empty tuple accumulates into all leaves
                allow_unreachable=True,
                accumulate_grad=True,
            )
        self.assertEqual(x_npu.device.type, device_type)
        self.assertEqual(x_npu.grad.cpu(), x_cpu.grad)

    def test_run_backward_return_grads(self):
        # Same call pattern as torch.autograd.grad: gradients are returned
        # instead of being accumulated into .grad.
        x = torch.randn(3, 3, device=device_type, requires_grad=True)
        y = (x * x + x).sum()
        grads = Variable._execution_engine.run_backward(
            (y,),
            (torch.ones_like(y),),
            False,  # retain_graph
            False,  # create_graph
            (x,),   # inputs: gradients are returned for them
            False,  # allow_unreachable
            accumulate_grad=False,
        )
        self.assertIsNone(x.grad)
        self.assertEqual(grads[0], 2 * x.detach() + 1)


if __name__ == "__main__":
    run_tests()
