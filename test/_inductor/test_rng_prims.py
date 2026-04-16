import torch
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from torch._prims.rng_prims import register_run_and_save_rng_state_op, run_and_save_rng_state
from testutils import TestUtils
import torch_npu


class TestRNGPrims(TestUtils):

    def test_default(self):
        register_run_and_save_rng_state_op()
        x = torch.randn(4, 4).to("npu")
        args = (x,) 
        kwargs = {}
        expected_rng_state = torch_npu.npu.get_rng_state()
        rng_state, out = run_and_save_rng_state(lambda x: x, *args, **kwargs)
        self.assertEqual(out, x)
        self.assertTrue(torch.equal(rng_state, expected_rng_state))


instantiate_parametrized_tests(TestRNGPrims)


if __name__ == "__main__":
    run_tests()
