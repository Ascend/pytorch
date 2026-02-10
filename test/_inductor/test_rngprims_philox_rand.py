import torch
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu


class TestRngprimsPhiloxRand(TestUtils):
    def op_calc(self, x):
        size = list(x.shape)

        seed = torch.tensor(1234, device=x.device, dtype=torch.int64)
        offset = torch.tensor(0, device=x.device, dtype=torch.int64)

        rand, new_offset = torch.ops.rngprims.philox_rand(
            size,           # SymInt[]
            seed,           # Tensor
            offset,         # Tensor
            None,           # stride
            x.device,       # device
            x.dtype,        # dtype
        )

        return rand * 2.0 + x

    @parametrize("shape", [(2, 4)])
    @parametrize("dtype", [torch.float32])
    def test_philox_rand_eager_vs_inductor(self, shape, dtype):
        device = "npu"

        x = torch.ones(shape, device=device, dtype=dtype)
        torch.manual_seed(0)
        y1_eager = self.op_calc(x)
        y2_eager = self.op_calc(x)

        compiled_op = torch.compile(
            self.op_calc,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )

        y1_ind = compiled_op(x)
        y2_ind = compiled_op(x)

        self.assertEqual(y1_eager, y2_eager)
        self.assertEqual(y1_ind, y2_ind)
        self.assertEqual(y1_eager, y1_ind)

instantiate_parametrized_tests(TestRngprimsPhiloxRand)

if __name__ == "__main__":
    run_tests()