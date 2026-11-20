import os

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401
import torch_npu.testing  # noqa: F401
from torch_npu.utils._dynamo import register_inductor_npu


register_inductor_npu()


# Pin log1p's pointwise path independently of the BCE decomposition: check
# values and ensure it does not silently return to an extern call.
class TestLog1pPointwise(TestCase):
    @torch._inductor.config.patch({
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
        "force_disable_caches": True,
        "implicit_fallbacks": False,
    })
    def test_log1p_pointwise(self, device):
        from torch._inductor.utils import run_and_get_code

        def log1p(x):
            return torch.log1p(x)

        compiled = torch.compile(log1p, fullgraph=True)
        for dtype, (atol, rtol) in (
            (torch.float16, (1e-3, 1e-3)),
            (torch.float32, (1e-5, 1.3e-5)),
        ):
            values = torch.tensor(
                [-0.999, -0.5, -1e-3, 0.0, 1e-3, 0.5, 1.0, 7.5, 40.0]
            ).to(device).to(dtype)
            actual, code = run_and_get_code(compiled, values)
            self.assertEqual(actual, log1p(values), atol=atol, rtol=rtol)
            for generated in code:
                statements = "\n".join(
                    line
                    for line in generated.splitlines()
                    if not line.lstrip().startswith("#")
                )
                self.assertNotIn("torch.ops.aten.log1p", statements)


instantiate_device_type_tests(TestLog1pPointwise, globals(), only_for=("npu",))

if __name__ == "__main__":
    run_tests()
