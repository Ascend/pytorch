import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import (
    run_tests,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu

torch._inductor.config.fallback_random = True


def dropout_with_backward(x):
    y = F.dropout(x, p=0.5, training=True)
    loss = y.sum()
    (grad_x,) = torch.autograd.grad(loss, x)
    return loss, grad_x


class TestDropoutCompile(TestUtils):
    def test_dropout_compile(self):
        device = "npu"

        torch.manual_seed(0)
        eager_x = torch.randn(4, 8, device=device, requires_grad=True)
        compiled_x = eager_x.detach().clone().requires_grad_(True)

        torch.manual_seed(42)
        eager_loss, eager_grad = dropout_with_backward(eager_x)

        torch.manual_seed(42)
        compiled_fn = torch.compile(dropout_with_backward, backend="inductor")
        compiled_loss, compiled_grad = compiled_fn(compiled_x)

        self.assertEqual(eager_loss, compiled_loss)
        self.assertEqual(eager_grad, compiled_grad)


instantiate_parametrized_tests(TestDropoutCompile)


if __name__ == "__main__":
    run_tests()