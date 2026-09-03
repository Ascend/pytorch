import functools
import unittest

import torch
import torch_npu
import torch_npu._inductor  # noqa: F401
from torch._inductor import metrics
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.flex_attention import flex_attention
from torch.testing import FileCheck
from torch_npu.testing.testcase import TestCase, run_tests


@unittest.skip("temporarily disabled due to known CI failure")
class TestFlexAttention(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        metrics.reset()

    def test_epilogue_fused(self):
        @torch.compile
        def f(q, k, v):
            return flex_attention(q, k, v).cos()

        q, k, v = (
            torch.randn(1, 8, 1024, 64, device="npu") for _ in range(3)
        )
        _, code = run_and_get_code(f, q, k, v)

        # FileCheck().check("triton_tem_fused").check_not("poi_fused_cos").run(
        #     code[0]
        # )
        accessed_bytes = 1 * 8 * 1024 * 64 * torch.float32.itemsize
        num_accesses = 6
        # TODO: Get rid of this fudge factor
        # We need this fudge factor for now as we write the extraneous logsumexp
        num_accesses += 1
        self.assertLess(metrics.num_bytes_accessed, accessed_bytes * num_accesses)

    def test_kernel_options_argument_is_respected(self):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 64),
            device="npu",
            dtype=torch.float32,
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        _, code = run_and_get_code(
            torch.compile(flex_attention),
            q,
            k,
            v,
            kernel_options={"BLOCK_M": 16},
        )

        FileCheck().check("BLOCK_M : tl.constexpr = 16").run(code[0])


if __name__ == "__main__":
    run_tests()
