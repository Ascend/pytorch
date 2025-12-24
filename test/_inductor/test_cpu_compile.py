import os
import torch
import numpy as np
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestNetworkCompile(TestUtils):
    @parametrize('input_dim', [4096])
    @parametrize('reshape_shape', [(1, 32, 1, 128)])
    @parametrize('device', ['cpu'])
    def test_network_compile_inference(self, input_dim, reshape_shape, device):
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, data1):
                relu_01 = self.relu(data1)
                reshape_01 = torch.reshape(relu_01, reshape_shape)
                softmax_01 = torch.nn.functional.softmax(reshape_01, dim=1)
                sqrt_01 = torch.sqrt(softmax_01)
                relu_02 = self.relu(sqrt_01)
                square_01 = torch.square(relu_02)
                add_01 = torch.add(square_01, square_01)
                return add_01


        torch.manual_seed(42)
        data1 = torch.randn(input_dim, device=device)

        model = Network().to(device)
        model.eval()

        compiled_model = torch.compile(model)

        with torch.no_grad():
            output = compiled_model(data1)
            cpu_out = output.detach().cpu().numpy()

        print(cpu_out)


instantiate_parametrized_tests(TestNetworkCompile)

if __name__ == "__main__":
    run_tests()