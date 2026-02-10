import torch
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu


class TestRunAndSaveRngState(TestUtils):
    def op_calc(self, like, device, dtype):
        rng_state1, res1 = torch._prims.rng_prims.run_and_save_rng_state(
            torch.ops.aten.rand_like.default,
            like,
            device=device,
            dtype=dtype,
        )

        torch_npu.npu.set_rng_state(rng_state1)

        rng_state2, res2 = torch._prims.rng_prims.run_and_save_rng_state(
            torch.ops.aten.rand_like.default,
            like,
            device=device,
            dtype=dtype,
        )

        return rng_state1, res1, rng_state2, res2

    @parametrize("shape", [(10,)])
    @parametrize("dtype", [torch.float32])
    def test_rng_state_with_compile(self, shape, dtype):
        device = "npu"

        like = torch.empty(shape, device=device, dtype=dtype)

        # eager
        rng_state1_eager, res1_eager, rng_state2_eager, res2_eager = \
            self.op_calc(like, device, dtype)

        self.assertEqual(res1_eager, res2_eager)
        self.assertTrue(torch.equal(rng_state1_eager, rng_state2_eager))                    

instantiate_parametrized_tests(TestRunAndSaveRngState)

if __name__ == "__main__":
    run_tests()