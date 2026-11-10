import os
import re
from unittest import mock

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
import torch_npu  # noqa: F401

from torch._inductor import config
from torch._inductor.graph import GraphLowering
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

PRIVATE_CALLS = (
    "torch.ops.npu.kl_div_backward.default(",
    "torch.ops.npu.kl_div_target_backward.default(",
)


class TestKlDivBackwardFallback(TestCase):
    def test_private_ops_are_decomposed_not_fallbacks(self):
        """The decomposition OWNS these ops; they must not also be fallbacks.

        make_fallback() raises "both a fallback and a decomp for same op", so
        listing them in EXPLICIT_FALLBACK_LIST while the decomposition is
        registered breaks backend load outright. #5792 is satisfied by the
        decomposition itself: the grad graph no longer carries an op that has
        neither a lowering nor a decomposition.
        """
        from torch._inductor.decomposition import select_decomp_table
        from torch_npu._inductor.triton_experimental.lowering_fallback_list import (
            EXPLICIT_FALLBACK_LIST,
        )

        # force the backend (and its decomposition registrar) to load
        torch._dynamo.reset()
        torch.compile(lambda a: a * 2 + 1,
                      options={"npu_backend": "triton_experimental"})(
            torch.ones(8, device="npu"))
        torch._dynamo.reset()

        table = select_decomp_table()
        for op in (torch.ops.npu.kl_div_backward.default,
                   torch.ops.npu.kl_div_target_backward.default):
            self.assertIn(op, table)
            self.assertNotIn(op, EXPLICIT_FALLBACK_LIST)

    @dtypes(torch.float16, torch.float32)
    @parametrize("reduction", ["none", "mean", "sum"])
    @parametrize("log_target", [False, True])
    def test_compiled_backward(self, device, dtype, reduction, log_target):
        input = torch.linspace(-0.8, 0.3, 6, device=device, dtype=dtype).reshape(2, 3).t().requires_grad_()
        target = torch.linspace(0.2, 0.9, 6, device=device, dtype=dtype).reshape(2, 3).t().requires_grad_()

        def fn(x, y):
            return torch.nn.functional.kl_div(x, y, reduction=reduction, log_target=log_target)

        expected = fn(input, target)
        expected_grads = torch.autograd.grad(expected.sum(), (input, target))
        generated = []
        with config.patch({
            "implicit_fallbacks": False,
            "test_configs.runtime_triton_shape_assert": True,
            "test_configs.runtime_triton_dtype_assert": True,
            "force_disable_caches": True,
        }), mock.patch.object(GraphLowering, "save_output_code", side_effect=generated.append):
            torch._dynamo.reset()
            actual = torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})(input, target)
            actual_grads = torch.autograd.grad(actual.sum(), (input, target))
        code = "\n".join(generated)

        # The backward is fused, not handed to the private ops: no aclnn call
        # survives, and a single generated kernel carries both gradients.
        for call in PRIVATE_CALLS:
            self.assertNotIn(call, code)
        fused = re.findall(r"def (triton_\w*kl_div_backward\w*kl_div_target_backward\w*)\(", code)
        self.assertTrue(fused, f"no fused KLDiv backward kernel in generated code:\n{code[:2000]}")

        self.assertEqual(actual, expected)
        self.assertEqual(actual_grads, expected_grads, atol=0, rtol=0)


instantiate_device_type_tests(TestKlDivBackwardFallback, globals(), only_for="npu")

if __name__ == "__main__":
    run_tests()
