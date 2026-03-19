import torch
from torch.testing._internal.common_utils import (
    run_tests,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu


class TestCurrentDevice(TestUtils):

    def test_npu_current_device(self):
        def fn(x):
            y = torch.empty(
                (2, 3), dtype=torch.float32, device=torch.npu.current_device()
            )
            y.copy_(x)
            return torch.sin(y + y.device.index)

        counter = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(backend=counter, fullgraph=True)(fn)

        with torch.npu.device(0):
            x = torch.randn(2, 3).to("npu")
            self.assertEqual(opt_fn(x), fn(x))
            self.assertEqual(counter.frame_count, 1)


instantiate_parametrized_tests(TestCurrentDevice)


if __name__ == "__main__":
    run_tests()