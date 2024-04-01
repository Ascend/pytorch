import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

# acl format
FORMAT_ND = 2
FORMAT_Z = 4
FORMAT_NZ = 29


class TestJitTrace(TestCase):

    def test_trace_npu_max(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x):
                x = torch_npu.npu_max(x, dim=1)
                return x

        example_input = torch.rand(2, 8).npu()
        model = NpuModel().to("npu")
        output1 = model(example_input)

        trace_model = torch.jit.trace(model, example_input)
        output2 = trace_model(example_input)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_silu_(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x):
                torch_npu.npu_silu_(x)
                return x

        example_input = torch.randn(5, 5).npu()
        example_input2 = example_input.clone()
        example_input3 = example_input.clone()

        model = NpuModel().to("npu")
        output1 = model(example_input)

        trace_model = torch.jit.trace(model, example_input2)
        output2 = trace_model(example_input3)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_slice_out(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x, out):
                torch_npu.npu_slice(x, [0, 0], [2, 2], out=out)
                return out

        example_input = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float16).npu()
        out_ = torch_npu.npu_slice(example_input, [0, 0], [2, 2])
        out1 = torch.zeros(out_.size(), dtype=out_.dtype).npu()
        out2 = out1.clone()
        out3 = out1.clone()

        model = NpuModel().to("npu")
        output1 = model(example_input, out1)

        trace_model = torch.jit.trace(model, (example_input, out2))
        output2 = trace_model(example_input, out3)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_bert_apply_adam_out(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, grad, var_in, m_in, v_in):
                max_grad_norm = -1.
                beta1 = 0.9
                beta2 = 0.99
                weight_decay = 0.
                lr = 0.
                epsilon = 1e-06
                global_grad_norm = 0.

                var_out, m_out, v_out = torch_npu.npu_bert_apply_adam(
                    lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay,
                    out=(var_in, m_in, v_in))
                return var_out, m_out, v_out

        seed = 3
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)

        var_in = torch.rand(321538).uniform_(-32., 21.).npu()
        m_in = torch.zeros(321538).npu()
        v_in = torch.zeros(321538).npu()
        grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()
        model = NpuModel().to("npu")
        output1 = model(grad, var_in, m_in, v_in)

        seed = 3
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)

        var_in = torch.rand(321538).uniform_(-32., 21.).npu()
        m_in = torch.zeros(321538).npu()
        v_in = torch.zeros(321538).npu()
        grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()
        trace_model = torch.jit.trace(model, (grad, var_in, m_in, v_in))

        seed = 3
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)

        var_in = torch.rand(321538).uniform_(-32., 21.).npu()
        m_in = torch.zeros(321538).npu()
        v_in = torch.zeros(321538).npu()
        grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()
        output2 = trace_model(grad, var_in, m_in, v_in)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_silu_and_npu_max(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x):
                torch_npu.npu_silu_(x)
                out = torch_npu.npu_max(x, dim=1)
                return out

        example_input = torch.randn(5, 5).npu()
        example_input2 = example_input.clone()
        example_input3 = example_input.clone()

        model = NpuModel().to("npu")
        output1 = model(example_input)

        trace_model = torch.jit.trace(model, example_input2)
        output2 = trace_model(example_input3)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_rotary_mul(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x, r1, r2):
                x = torch_npu.npu_rotary_mul(x, r1, r2)
                return x

        x = torch.rand([8192, 2, 5, 128], dtype=torch.float32).npu()
        r1 = torch.rand([8192, 1, 1, 128], dtype=torch.float32).npu()
        r2 = torch.rand([8192, 1, 1, 128], dtype=torch.float32).npu()
        model = NpuModel().to("npu")
        output1 = model(x, r1, r2)

        trace_model = torch.jit.trace(model, (x, r1, r2))
        output2 = trace_model(x, r1, r2)
        self.assertRtolEqual(output1, output2)


if __name__ == "__main__":
    run_tests()
