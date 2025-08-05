import torch
from torch.testing._internal.common_utils import run_tests
from testutils import TestUtils
import torch_npu


class TestLazyRegister(TestUtils):
    def test_compile_but_not_invoked(self):

        def run(x, y):
            return x + y

        run = torch.compile(run)
        self.assertFalse(torch_npu.utils._dynamo.is_inductor_npu_initialized())
    
    def test_disale_register_inductor_npu(self):
        torch_npu.utils._dynamo.disable_register_inductor_npu()

        def run(x, y):
            return x - y

        run = torch.compile(run)
        x = torch.randn(10, 20).npu()
        y = torch.randn(10, 20).npu()

        with self.assertRaisesRegex(Exception, "Device npu not supported"):
            _ = run(x, y)

        self.assertFalse(torch_npu.utils._dynamo.is_inductor_npu_initialized())

        torch_npu.utils._dynamo.enable_register_inductor_npu()
    
    def test_no_register_for_cpu(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.batch_norm = torch.nn.BatchNorm2d(256)

            def forward(self, input_tensor):
                reshaped_tensor = input_tensor.view(128, 192, 1, 256)

                bn_input = reshaped_tensor.permute(
                    0, 3, 1, 2
                )
                batch_norm_tensor = self.batch_norm(bn_input)

                reshaped_bn_tensor = batch_norm_tensor.permute(0, 2, 3, 1).view(128, 192, 256)

                constant_tensor = torch.ones_like(reshaped_bn_tensor)

                sub_output = reshaped_bn_tensor - constant_tensor
                final_result = sub_output.view(128, 192, 256)

                return final_result
        
        model = Model()
        compiled_model = torch.compile(model)
        input_tensor = torch.randn(128, 192, 256)

        output = model(input_tensor)
        output1 = compiled_model(input_tensor)

        self.assertFalse(torch_npu.utils._dynamo.is_inductor_npu_initialized())
        self.assertEqual(output, output1, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    run_tests()
