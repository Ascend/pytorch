import torch
import torch._dynamo.test_case
from torch._dynamo.testing import same

import torch_npu  # noqa: F401
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

    def test_multithreaded_cudagraph_trees(self):
        # A single compiled callable may be first invoked from multiple Python
        # threads. Graph capture and private-pool warmup are process-wide states,
        # so callables/managers must be isolated per thread while warmup and
        # capture are serialized. See q4_feat #1 Python 多线程 Graph Tree.
        import queue
        import threading
        import traceback

        _graph_tree.reset_npugraph_trees()
        self.addCleanup(_graph_tree.reset_npugraph_trees)

        def foo(x, y):
            return torch.sin(x) + torch.cos(y)

        opt_foo = torch.compile(foo, mode="reduce-overhead")
        num_threads = 3
        barrier = threading.Barrier(num_threads)
        errors = queue.Queue()
        results = queue.Queue()

        def run_once(check_manager, start_barrier=None):
            try:
                x = torch.rand(4, 4, device="npu")
                y = torch.rand(4, 4, device="npu")
                expected = foo(x, y)
                if start_barrier is not None:
                    start_barrier.wait()
                opt_foo(x, y)
                result = opt_foo(x, y)
                if check_manager:
                    manager = _graph_tree.get_manager(
                        x.device.index, create_if_none_exists=False
                    )
                    self.assertIsNotNone(manager)
                results.put((result, expected))
            except Exception:
                errors.put(traceback.format_exc())

        # First, warm up from a non-main thread to ensure compilation works.
        owner_thread = threading.Thread(target=run_once, args=(True,))
        owner_thread.start()
        owner_thread.join()

        def check_errors():
            if not errors.empty():
                messages = []
                while not errors.empty():
                    messages.append(errors.get())
                self.fail("\n".join(messages))

        check_errors()
        result, expected = results.get()
        self.assertTrue(same(result, expected))
        del result, expected

        # Then invoke concurrently from several threads, with a barrier to
        # maximize interleaving of the capture paths.
        def worker():
            run_once(True, barrier)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        check_errors()
        while not results.empty():
            result, expected = results.get()
            self.assertTrue(same(result, expected))
            del result, expected

    def test_cond_tree_warmup(self):
        # torch.cond goes through tree warmup before graph capture; the warmup
        # path must use ControlFlowOpWarmupDispatchMode to simulate conditional
        # capture so eager warmup matches the final recording.
        _graph_tree.reset_npugraph_trees()
        self.addCleanup(_graph_tree.reset_npugraph_trees)

        def fn(x):
            # Use element indexing for the scalar predicate; avoid reductions so
            # the test exercises torch.cond warmup, not reduction lowering.
            return torch.cond(x[0, 0] > 0, lambda: x + 1, lambda: x - 1, ())

        x = torch.randn(4, 4, device="npu")
        compiled = torch.compile(fn, mode="reduce-overhead")
        # Exercise tree warmup, capture and replay. Flip the predicate in
        # place (same storage, so the input does not trigger a re-record):
        # a capture that hard-codes the branch taken at capture time would
        # return the wrong branch when it is replayed after the flip.
        self.assertTrue(same(compiled(x), fn(x)))
        for value in (1.0, -1.0, 1.0):
            x.fill_(value)
            self.assertTrue(same(compiled(x), fn(x)))

    @torch._dynamo.config.patch("error_on_recompile", True)
    @torch._inductor.config.patch("triton.cudagraph_unexpected_rerecord_limit", 1)
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
            # Otherwise the tree stays in the recording state and the new
            # parameter pointer is never checked against the existing roots.
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
        # Counter never bumped for any (node, function).
        for fn_to_count in manager.num_rerecord.values():
            for count in fn_to_count.values():
                self.assertEqual(count, 0)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
