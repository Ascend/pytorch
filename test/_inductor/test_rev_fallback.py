import unittest
from statistics import median

import torch
import torch_npu  # noqa: F401 - register the NPU device backend
from torch._inductor import config
from torch._inductor.utils import run_and_get_code

from torch_npu._inductor import config as npu_config


def rev_slice_clone_reshape(x, stop):
    reversed_x = torch.ops.prims.rev.default(x, [1])
    sliced = torch.ops.aten.slice.Tensor(reversed_x, 1, 0, stop)
    return sliced.clone(memory_format=torch.contiguous_format).reshape(-1, x.size(-1))


@unittest.skipUnless(torch.npu.is_available(), "NPU is required")
class TestRevFallback(unittest.TestCase):
    def test_dynamic_rev_slice_clone_reshape_falls_back(self):
        if not torch.npu.is_bf16_supported():
            self.skipTest("BF16 is required")

        inputs = [
            torch.randn(batch, 513, 256, dtype=torch.bfloat16, device="npu")
            for batch in (4, 3)
        ]
        for x in inputs:
            torch._dynamo.mark_dynamic(x, 0)

        with config.patch(cpp_wrapper=False, compile_threads=1, force_disable_caches=True):
            compiled = torch.compile(rev_slice_clone_reshape, fullgraph=True)
            actual, code = run_and_get_code(compiled, inputs[0], 480)
            torch.testing.assert_close(
                actual, rev_slice_clone_reshape(inputs[0], 480), rtol=0, atol=0, check_stride=True
            )
            self.assertIn("torch.ops.prims.rev.default(", "\n".join(code))
            torch.testing.assert_close(
                compiled(inputs[1], 480), rev_slice_clone_reshape(inputs[1], 480),
                rtol=0, atol=0, check_stride=True,
            )

    @unittest.skipUnless(npu_config.is_ascend950, "performance regression requires Ascend 950")
    def test_dynamic_rev_slice_clone_reshape_performance(self):
        x = torch.randn(4, 100_000, 256, dtype=torch.bfloat16, device="npu")
        torch._dynamo.mark_dynamic(x, 0)
        stop = 95_904

        def eager(value, stop):
            # A consumer keeps slice/clone fusion useful after rev falls back.
            return rev_slice_clone_reshape(value, stop).relu()

        def time_ms(fn):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(3):
                fn(x, stop)
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / 3

        with config.patch(
            cpp_wrapper=False,
            compile_threads=1,
            force_disable_caches=True,
            max_autotune=True,
            coordinate_descent_tuning=True,
        ):
            compiled = torch.compile(eager, fullgraph=True)
            torch.testing.assert_close(compiled(x, stop), eager(x, stop), rtol=0, atol=0)
            for _ in range(3):
                eager(x, stop)
                compiled(x, stop)
            torch.npu.synchronize()

            samples = [(time_ms(eager), time_ms(compiled)) for _ in range(5)]

        eager_ms = median(elapsed for elapsed, _ in samples)
        compiled_ms = median(elapsed for _, elapsed in samples)
        timings = f"rev slice clone reshape relu: eager {eager_ms:.3f} ms, compiled {compiled_ms:.3f} ms"
        print(f"\n{timings}")
        self.assertLess(compiled_ms, eager_ms, timings)


if __name__ == "__main__":
    unittest.main()
