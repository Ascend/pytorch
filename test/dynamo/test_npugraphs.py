import torch
import torch._dynamo.test_case
from torch._dynamo.testing import same
from torch._inductor import config as inductor_config

import torch_npu
import torch_npu.npu._graph_tree as _graph_tree


class NpuGraphsTests(torch._dynamo.test_case.TestCase):
    def test_inference_mode_does_not_need_mark_step_begin(self):
        # This tests that torch.inference_mode() removes the need for
        # the user to call torch.compiler.mark_step_begin()

        # reset to make sure that warned_functions is empty
        _graph_tree.reset_npugraph_trees()
        self.addCleanup(_graph_tree.reset_npugraph_trees)

        def foo(x):
            with torch.inference_mode():
                return x.sin()

        x = torch.randn(4, device="npu", requires_grad=False)

        compiled_foo = torch.compile(foo, backend="npugraphs")

        # First call runs eager, second captures and replays, rest
        # should just replay.
        for _ in range(4):
            out = compiled_foo(x)
            self.assertTrue(same(out, foo(x)))

        manager = _graph_tree.get_manager(x.device.index)
        self.assertTrue(
            len(manager.warned_functions) == 0,
            "Replaying inference workload should not warn about repeated graph captures",
        )

    @torch._dynamo.config.patch("error_on_recompile", True)
    @inductor_config.patch("triton.cudagraph_unexpected_rerecord_limit", 1)
    def test_param_ptr_churn_does_not_count_toward_rerecord_limit(self):
        # Distinct nn.Parameter pointers across calls trigger StaticInputIdxMismatch
        # each time. Those mismatches re-record but must not count against
        # cudagraph_unexpected_rerecord_limit.
        _graph_tree.reset_npugraph_trees()
        self.addCleanup(_graph_tree.reset_npugraph_trees)

        def fn(x, y):
            return x * y

        fn_compiled = torch.compile(fn, mode="reduce-overhead")
        # Burn through more distinct params than the limit (1) allows if counted.
        for _ in range(5):
            p = torch.nn.Parameter(torch.rand(2, 2, device="npu"))
            expected = fn(p, p)
            # No backward() is invoked, so explicitly end each invocation.
            torch.compiler.npugraph_mark_step_begin()
            result = fn_compiled(p, p)
            self.assertTrue(same(result, expected))
            torch.compiler.npugraph_mark_step_begin()
            result = fn_compiled(p, p)
            self.assertTrue(same(result, expected))

        manager = _graph_tree.get_manager(0, create_if_none_exists=False)
        self.assertIsNotNone(manager)
        # Confirm that re-recording actually happened, otherwise the zero
        # counters below would not prove that churn is excluded from the limit.
        self.assertGreater(manager.debug_fail_counter, 0)
        for fn_to_count in manager.num_rerecord.values():
            for count in fn_to_count.values():
                self.assertEqual(count, 0)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()