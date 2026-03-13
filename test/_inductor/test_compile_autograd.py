import os
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor


class DeviceCheckFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # 在 backward 中访问 device 对象
        dev = torch.device("npu")
        return grad_output


class TestCompiledAutograd(TestUtils):

    @parametrize("input_dim", [10])
    @parametrize("device", ["npu"])
    def test_compiled_autograd(self, input_dim, device):

        def model_fn(x):
            return DeviceCheckFunc.apply(x).sum()

        torch.manual_seed(42)

        x = torch.randn(input_dim, requires_grad=True, device=device)

        with torch._dynamo.utils.maybe_enable_compiled_autograd(True):
            compiled_fn = torch.compile(
                model_fn,
                backend="inductor",
                dynamic=False
            )

            loss = compiled_fn(x)
            loss.backward()

        grad = x.grad.detach().cpu().numpy()
        print("grad:", grad)


instantiate_parametrized_tests(TestCompiledAutograd)


if __name__ == "__main__":
    run_tests()