import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu


class TestEvent(TestUtils):
    def op_calc(self, x, y):
        x = x * y
        event = torch.npu.Event()
        event.record()
        out = x + y
        return out

    @parametrize("shape", [(2, 2)])
    def test_npu_event_in_graph(self, shape):
        device = torch.device("npu")
        x = torch.randn(shape, device=device, dtype=torch.float32)
        y = torch.randn(shape, device=device, dtype=torch.float32)
        std_out = self.op_calc(x, y)
        compiled_model = torch.compile(self.op_calc, backend="inductor")
        compile_out, codes = run_and_get_code(compiled_model, x, y)
        self.assertEqual(std_out, compile_out, atol=1e-5, rtol=1e-5)
        self.assertTrue("triton_unk_fused_add_mul_0.run" in codes[0])


instantiate_parametrized_tests(TestEvent)

if __name__ == "__main__":
    run_tests()
