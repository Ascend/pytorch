import torch
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

    def test_gemm_lowerings_are_device_dispatched(self):
        from torch._inductor import lowering

        for target in (
            torch.ops.aten.mm.default,
            torch.ops.aten.addmm.default,
            torch.ops.aten.bmm.default,
        ):
            with self.subTest(target=target):
                handler = lowering.lowerings[target]
                self.assertTrue(
                    getattr(
                        handler,
                        "_torch_npu_device_lowering_dispatch",
                        False,
                    )
                )

    def test_cpu_gemm_compile_after_npu_registration(self):
        cases = (
            (
                "mm",
                lambda mat1, mat2: torch.mm(mat1, mat2),
                (torch.randn(4, 5), torch.randn(5, 6)),
            ),
            (
                "addmm",
                lambda bias, mat1, mat2: torch.addmm(bias, mat1, mat2),
                (torch.randn(4, 6), torch.randn(4, 5), torch.randn(5, 6)),
            ),
            (
                "bmm",
                lambda mat1, mat2: torch.bmm(mat1, mat2),
                (torch.randn(2, 4, 5), torch.randn(2, 5, 6)),
            ),
        )

        for name, fn, args in cases:
            with self.subTest(name=name):
                torch._dynamo.reset()
                compiled = torch.compile(fn, fullgraph=True)
                self.assertEqual(compiled(*args), fn(*args))


instantiate_parametrized_tests(TestNetworkCompile)

if __name__ == "__main__":
    run_tests()
