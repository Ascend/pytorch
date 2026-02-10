import torch
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu


class TestRunWithRngState(TestUtils):
    def op_calc(self, current_state, like, device, dtype):
        res1 = torch._prims.rng_prims.run_with_rng_state(
            current_state,
            torch.ops.aten.rand_like.default,
            like,
            device=device,
            dtype=dtype,
        )

        res2 = torch._prims.rng_prims.run_with_rng_state(
            current_state,
            torch.ops.aten.rand_like.default,
            like,
            device=device,
            dtype=dtype,
        )

        return res1, res2

    @parametrize("shape", [(10,)])
    @parametrize("dtype", [torch.float32])
    def test_rng_state_with_compile(self, shape, dtype):
        device = "npu"
        torch.manual_seed(0)
        current_state = torch_npu.npu.get_rng_state()
        like = torch.empty(shape, device=device, dtype=dtype)

        # eager
        res1_eager, res2_eager = \
            self.op_calc(current_state, like, device, dtype)

        # compiled
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        res1_ind, res2_ind = \
            compiled_op_calc(current_state, like, device, dtype)

        self.assertEqual(res1_eager, res2_eager)
        self.assertEqual(res1_ind, res2_ind)
        self.assertEqual(res1_ind, res1_eager)

instantiate_parametrized_tests(TestRunWithRngState)

if __name__ == "__main__":
    run_tests()