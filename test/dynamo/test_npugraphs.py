import torch
import torch._dynamo.test_case
from torch._dynamo.testing import same

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

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()