import os

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests

import torch_npu  # noqa: F401
from torch_npu.utils._dynamo import register_inductor_npu


register_inductor_npu()


class TestBatchNormSavedStats(TestCase):
    @parametrize("dtype,stats_dtype", [
        (torch.float16, torch.float16),
        (torch.float16, torch.float32),
        (torch.float32, torch.float32),
    ])
    @parametrize("functional", [False, True])
    @torch._inductor.config.patch({
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
        "force_disable_caches": True,
    })
    def test_update_saved_statistics(self, device, dtype, stats_dtype, functional):
        torch.manual_seed(123)
        values = [
            torch.randn(5, 5, 5) * 3,
            torch.linspace(0.5, 1.5, 5),
            torch.linspace(-0.25, 0.25, 5),
            torch.linspace(-0.4, 0.4, 5),
            torch.linspace(0.8, 1.2, 5),
        ]
        eager_args = [
            value.to(device=device, dtype=dtype if i == 0 else stats_dtype)
            for i, value in enumerate(values)
        ]
        compiled_args = [value.clone() for value in eager_args]
        for args in (eager_args, compiled_args):
            for value in args[:3]:
                value.requires_grad_(True)

        op = (
            torch.ops.aten._batch_norm_with_update_functional.default
            if functional
            else torch.ops.aten._batch_norm_with_update.default
        )

        def fn(*args):
            return op(*args, 0.5, 0.6)

        expected = fn(*eager_args)
        actual = torch.compile(fn, fullgraph=True)(*compiled_args)
        atol, rtol = (1e-5, 1e-3) if dtype == torch.float16 else (1.5e-5, 1.3e-5)
        self.assertEqual(actual[:3], expected[:3], atol=atol, rtol=rtol)
        self.assertEqual(actual[3:], expected[3:])
        self.assertEqual(compiled_args[3:], eager_args[3:])
        grad = torch.linspace(-1, 1, 125, device=device, dtype=dtype).reshape(5, 5, 5)
        expected[0].backward(grad)
        actual[0].backward(grad)
        for eager, compiled in zip(eager_args[:3], compiled_args[:3]):
            self.assertEqual(compiled.grad, eager.grad, atol=atol, rtol=rtol)


instantiate_device_type_tests(TestBatchNormSavedStats, globals(), only_for=("npu",))


if __name__ == "__main__":
    run_tests()
