# Owner(s): ["module: dynamo"]
"""Dual-stream graph replay over two isolated native graph trees (NPU).

Contract tests for ``torch_npu._experimental.dual_stream_graph``:

- two lanes own two independent native ``TreeManagerContainer`` /
  ``NPUGraphTreeManager`` instances, each with its own private memory pool;
- one compiled artifact is shared by both lanes (backend identity), while
  the per-domain graphify caches keep the lane graph state apart;
- the graph tree core is never edited: without an active domain the
  original container lookup / reset / compile bindings behave as baseline.

Model choice mirrors the previous dual-stream tests: a small two-input MLP
with relu (gelu hits a torch_npu fallback registration collision on this
stack, unrelated to dual stream).  The runner accepts a raw function, so the
model is wrapped in a plain ``def region(x, y)``.
"""

import sys
import unittest
import weakref
from pathlib import Path

import torch
import torch_npu  # noqa: F401  (import before touching torch_npu.__path__)

# Load the experimental package from this repo while torch_npu itself keeps
# resolving to the installed (overlay) package: the repo torch_npu directory
# is appended AFTER the installed path entry, so it can only provide
# subpackages missing from the installed tree.
_REPO_TORCH_NPU = str(Path(__file__).resolve().parents[2] / "torch_npu")
if _REPO_TORCH_NPU not in torch_npu.__path__:
    torch_npu.__path__.append(_REPO_TORCH_NPU)

from torch_npu._experimental.dual_stream_graph import (  # noqa: E402
    DualBatchRunner,
    GraphExecutionDomain,
    current_domain,
    reset_graph_execution_domains,
)
from torch_npu.npu import _graph_tree  # noqa: E402
from torch_npu.npu._graph_resource_pool import GraphResourcePool  # noqa: E402


NPU_DEV = "npu"


class DualInputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 8)

    def forward(self, x, y):
        return self.fc2(torch.nn.functional.relu(self.fc1(x + y)))


def region(model, x, y):
    return model(x, y)


def _tree_stats(manager):
    """Capture-observable counters of one native tree manager."""
    return (
        len(manager.ids_to_funcs),
        sum(len(nodes) for nodes in manager.roots.values()),
        len(manager.warmed_up_functions),
    )


@unittest.skipIf(not torch.npu.is_available(), "requires NPU")
class DualStreamGraphTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        cls.model = DualInputModel().to(NPU_DEV).eval()

    def setUp(self):
        self._domains = []

    def tearDown(self):
        for domain in self._domains:
            domain.close()
        self._domains.clear()
        # Every DualBatchRunner owns a distinct Dynamo backend identity (by
        # design, so compiled code is never shared with an ordinary compile),
        # which means each test recompiles the same frames afresh.  Dynamo
        # counts recompiles per code object and, once
        # `torch._dynamo.config.recompile_limit` (default 8) is exceeded,
        # silently falls back to eager for that frame -- results stay correct
        # but nothing is captured and native containers are never created.
        # Clearing Dynamo state per test keeps each test's recompile budget
        # independent, exactly as a fresh process would.
        torch._dynamo.reset()

    def captured_manager(self, runner, lane):
        """Return lane's native manager, failing loudly if no capture happened.

        A silent Dynamo fallback to eager is the failure mode this guards
        against: correctness assertions alone cannot distinguish it.
        """
        container = runner.domains[lane].native_container
        self.assertIsNotNone(
            container,
            f"lane {lane} never created a native container: the graph was not "
            "captured (Dynamo most likely fell back to eager -- check "
            "config.recompile_limit and guards)",
        )
        manager = container.tree_manager
        self.assertIsNotNone(manager, f"lane {lane} container has no manager")
        return manager

    def make_runner(self, lane_selector=None):
        runner = DualBatchRunner(
            lambda x, y: region(self.model, x, y),
            batch_independent=True,
            dynamic=True,
            options={"triton.cudagraphs": True},
            lane_selector=lane_selector,
        )
        self.addCleanup(runner.close)
        return runner

    def run_split(self, runner, x, y, recover=lambda outs: torch.cat(outs, dim=0)):
        split = x.shape[0] // 2
        parts = [(x[:split].contiguous(), y[:split].contiguous()),
                 (x[split:].contiguous(), y[split:].contiguous())]
        groups = [((px, py), {}) for px, py in parts]
        return runner.run(groups, recover)

    def expect_close(self, actual, expected, msg=""):
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4, msg=msg)


class TestTwoDomainIsolation(DualStreamGraphTestBase):
    def test_two_lanes_two_native_managers_and_pools(self):
        runner = self.make_runner()
        x = torch.randn(72, 64, device=NPU_DEV)
        y = torch.randn(72, 64, device=NPU_DEV)
        with torch.no_grad():
            result = self.run_split(runner, x, y)
        self.expect_close(result, region(self.model, x, y))

        domain0, domain1 = runner.domains
        manager0 = self.captured_manager(runner, 0)
        manager1 = self.captured_manager(runner, 1)
        self.assertIsNot(domain0.native_container, domain1.native_container)
        self.assertIsNot(manager0, manager1)
        # each native manager owns its private pool
        self.assertNotEqual(
            manager0.npu_graphs_thread_pool, manager1.npu_graphs_thread_pool
        )
        # per-domain graphify caches are separate objects with one artifact each
        self.assertEqual(len(domain0._callables), 1)
        self.assertEqual(len(domain1._callables), 1)
        self.assertIsNot(domain0._callables, domain1._callables)
        # one compiled artifact shared by both lanes
        self.assertEqual(runner.compile_count, 1)

    def test_steady_state_does_not_recapture(self):
        runner = self.make_runner()
        x = torch.randn(72, 64, device=NPU_DEV)
        y = torch.randn(72, 64, device=NPU_DEV)
        with torch.no_grad():
            first = self.run_split(runner, x, y)
            # warmup + record may span the first invocations; run until stable
            for _ in range(3):
                self.run_split(runner, x, y)
        stats0 = _tree_stats(self.captured_manager(runner, 0))
        stats1 = _tree_stats(self.captured_manager(runner, 1))
        with torch.no_grad():
            again = self.run_split(runner, x, y)
        self.assertEqual(_tree_stats(self.captured_manager(runner, 0)), stats0)
        self.assertEqual(_tree_stats(self.captured_manager(runner, 1)), stats1)
        self.expect_close(again, first)

    def test_single_gear_warming_reused_in_combination(self):
        # Warm 48 on lane 0 and 8 on lane 1 as separate requests, then run the
        # 56 = 48 + 8 combination: both lanes must hit their warmed graph
        # instead of capturing new combined-region graphs.
        def selector(groups):
            if len(groups) == 1:
                (args, _), = groups
                return [1 if args[0].shape[0] < 16 else 0]
            return [0, 1]

        runner = self.make_runner(lane_selector=selector)
        x = torch.randn(56, 64, device=NPU_DEV)
        y = torch.randn(56, 64, device=NPU_DEV)
        with torch.no_grad():
            runner.run([((x[:48].contiguous(), y[:48].contiguous()), {})], lambda o: o[0])
            runner.run([((x[48:].contiguous(), y[48:].contiguous()), {})], lambda o: o[0])
            for _ in range(3):
                runner.run(
                    [
                        ((x[:48].contiguous(), y[:48].contiguous()), {}),
                        ((x[48:].contiguous(), y[48:].contiguous()), {}),
                    ],
                    lambda o: torch.cat(o, dim=0),
                )
        stats0 = _tree_stats(self.captured_manager(runner, 0))
        stats1 = _tree_stats(self.captured_manager(runner, 1))
        with torch.no_grad():
            result = runner.run(
                [
                    ((x[:48].contiguous(), y[:48].contiguous()), {}),
                    ((x[48:].contiguous(), y[48:].contiguous()), {}),
                ],
                lambda o: torch.cat(o, dim=0),
            )
        self.assertEqual(_tree_stats(self.captured_manager(runner, 0)), stats0)
        self.assertEqual(_tree_stats(self.captured_manager(runner, 1)), stats1)
        self.expect_close(result, region(self.model, x, y))

    def test_lane_selector_routes_single_gear(self):
        seen = []

        def selector(groups):
            seen.append(len(groups))
            # single-chunk requests warm lane 1; pairs use lanes in order
            return [1] if len(groups) == 1 else [0, 1]

        runner = self.make_runner(lane_selector=selector)
        x = torch.randn(48, 64, device=NPU_DEV)
        y = torch.randn(48, 64, device=NPU_DEV)
        with torch.no_grad():
            self.run_split(runner, x, y)
        self.assertEqual(seen[-1], 2)
        self.assertEqual(len(self.captured_manager(runner, 1).ids_to_funcs), 1)
        self.assertEqual(len(self.captured_manager(runner, 0).ids_to_funcs), 1)


class TestDefaultPathUnaffected(DualStreamGraphTestBase):
    def test_no_active_domain_keeps_baseline_container(self):
        self.assertIsNone(current_domain())
        default_container = _graph_tree.get_container(0)
        # the wrapped lookup still serves the TLS container without a domain
        self.assertIs(_graph_tree.get_container(0), default_container)
        self.assertIn(0, _graph_tree.get_obj(
            _graph_tree.local, "npu_tree_manager_containers"
        ))

    def test_domain_activation_routes_container_and_restores(self):
        domain = GraphExecutionDomain(0)
        self._domains.append(domain)
        with domain.activate():
            self.assertIs(current_domain(), domain)
            domain_container = domain.container_for(0)
            self.assertIs(_graph_tree.get_container(0), domain_container)
        self.assertIsNone(current_domain())
        # after deactivation the default TLS path is served again
        self.assertIsNot(_graph_tree.get_container(0), domain_container)

    def test_custom_backend_compilation_uses_experimental_entry(self):
        # activating a domain installs the routed cudagraphify binding once
        domain = GraphExecutionDomain(0)
        self._domains.append(domain)
        from torch._inductor import compile_fx

        with domain.activate():
            pass
        # the routed entry stays installed process-wide after activation
        self.assertEqual(
            compile_fx.cudagraphify.__name__, "_inductor_routed_npugraphify"
        )
        # the core container lookup is wrapped exactly once
        self.assertTrue(
            getattr(_graph_tree.get_container, "_routes_npu_execution_domains", False)
        )
        self.assertEqual(
            _graph_tree.get_container.__wrapped__.__name__, "get_container"
        )


class TestRejections(DualStreamGraphTestBase):
    def test_reject_without_batch_independent(self):
        with self.assertRaisesRegex(ValueError, "batch_independent"):
            DualBatchRunner(lambda x: x)

    def test_reject_compiled_wrapper(self):
        compiled = torch.compile(lambda x: x)
        with self.assertRaisesRegex(ValueError, "raw function"):
            DualBatchRunner(compiled, batch_independent=True)

    def test_reject_module_directly(self):
        with self.assertRaisesRegex(ValueError, "raw"):
            DualBatchRunner(DualInputModel(), batch_independent=True)

    def test_reject_grad_mode(self):
        runner = self.make_runner()
        x = torch.randn(8, 64, device=NPU_DEV)
        y = torch.randn(8, 64, device=NPU_DEV)
        with self.assertRaisesRegex(RuntimeError, "no_grad"):
            runner.run([((x, y), {})], lambda o: o[0])

    def test_reject_duplicate_lanes(self):
        runner = self.make_runner(lane_selector=lambda groups: [0, 0])
        x = torch.randn(72, 64, device=NPU_DEV)
        y = torch.randn(72, 64, device=NPU_DEV)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "different lanes"):
            self.run_split(runner, x, y)

    def test_reject_three_chunks(self):
        runner = self.make_runner()
        x = torch.randn(72, 64, device=NPU_DEV)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "one or two"):
            runner.run([((x,), {})] * 3, lambda o: o[0])

    def test_reject_cross_device(self):
        if torch.npu.device_count() < 2:
            self.skipTest("requires 2 NPUs")
        runner = self.make_runner()
        x0 = torch.randn(8, 64, device="npu:0")
        y0 = torch.randn(8, 64, device="npu:0")
        x1 = torch.randn(8, 64, device="npu:1")
        y1 = torch.randn(8, 64, device="npu:1")
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "span devices"):
            runner.run([((x0, y0), {}), ((x1, y1), {})], lambda o: torch.cat(o, dim=0))

    def test_reject_nested_run(self):
        runner = self.make_runner()
        x = torch.randn(8, 64, device=NPU_DEV)
        y = torch.randn(8, 64, device=NPU_DEV)
        inner_started = []

        def nested_recover(outs):
            inner_started.append(True)
            with self.assertRaisesRegex(RuntimeError, "reentrant"):
                runner.run([((x, y), {})], lambda o: o[0])
            return torch.cat(outs, dim=0)

        with torch.no_grad():
            runner.run([((x, y), {})], nested_recover)
        self.assertTrue(inner_started)

    def test_reject_adaptive_pool_active(self):
        pool = GraphResourcePool.get_pool(0)
        pool.activate()
        try:
            runner = self.make_runner()
            x = torch.randn(8, 64, device=NPU_DEV)
            y = torch.randn(8, 64, device=NPU_DEV)
            with torch.no_grad(), self.assertRaisesRegex(
                RuntimeError, "adaptive"
            ):
                runner.run([((x, y), {})], lambda o: o[0])
        finally:
            pool.deactivate()


class TestLifecycle(DualStreamGraphTestBase):
    def test_exception_drains_and_next_run_recovers(self):
        runner = self.make_runner()
        x = torch.randn(72, 64, device=NPU_DEV)
        y = torch.randn(72, 64, device=NPU_DEV)

        class Boom(torch.nn.Module):
            def forward(self, x, y):
                raise ValueError("lane failure")

        boom = Boom().to(NPU_DEV).eval()
        split = x.shape[0] // 2
        groups = [
            ((x[:split].contiguous(), y[:split].contiguous()), {}),
            ((x[split:].contiguous(), y[split:].contiguous()), {}),
        ]
        original_model = self.model
        self.model = boom
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "lane failure"):
            runner.run(groups, lambda o: torch.cat(o, dim=0))
        self.assertIsNone(current_domain())
        self.model = original_model
        with torch.no_grad():
            result = runner.run(groups, lambda o: torch.cat(o, dim=0))
        self.expect_close(result, region(original_model, x, y))

    def test_close_is_idempotent_and_run_rejected_after(self):
        runner = self.make_runner()
        runner.close()
        runner.close()
        x = torch.randn(8, 64, device=NPU_DEV)
        y = torch.randn(8, 64, device=NPU_DEV)
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "closed"):
            runner.run([((x, y), {})], lambda o: o[0])

    def test_global_reset_coordinated_with_domains(self):
        runner = self.make_runner()
        x = torch.randn(72, 64, device=NPU_DEV)
        y = torch.randn(72, 64, device=NPU_DEV)
        with torch.no_grad():
            self.run_split(runner, x, y)
        epoch0 = runner.domains[0].epoch
        reset_graph_execution_domains()
        self.assertEqual(runner.domains[0].epoch, epoch0 + 1)
        self.assertEqual(runner.domains[1].epoch, epoch0 + 1)
        self.assertEqual(len(runner.domains[0]._callables), 0)
        # a following run rebuilds cleanly
        with torch.no_grad():
            result = self.run_split(runner, x, y)
        self.expect_close(result, region(self.model, x, y))

    def test_reset_rejected_while_active(self):
        domain = GraphExecutionDomain(0)
        self._domains.append(domain)
        with domain.activate(), self.assertRaisesRegex(RuntimeError, "active"):
            reset_graph_execution_domains()

    def test_inflight_references_kept_until_completion(self):
        runner = self.make_runner()
        x = torch.randn(72, 64, device=NPU_DEV)
        y = torch.randn(72, 64, device=NPU_DEV)
        with torch.no_grad():
            result = self.run_split(runner, x, y)
        ref = weakref.ref(result)
        del result
        # references are retained by the domain until the completion event;
        # after close() the event has been waited and storage may be freed.
        runner.close()
        self.assertIsNone(ref())


class TestOutputContract(DualStreamGraphTestBase):
    def test_recover_may_return_lane_clone_directly(self):
        runner = self.make_runner()
        x = torch.randn(72, 64, device=NPU_DEV)
        y = torch.randn(72, 64, device=NPU_DEV)
        with torch.no_grad():
            # returning lane 0's clone (a view of it) keeps a private
            # allocation stream; record_stream on the caller must cover it.
            result = self.run_split(runner, x, y, recover=lambda outs: outs[0])
        expected = region(self.model, x[:36].contiguous(), y[:36].contiguous())
        self.expect_close(result, expected)

    def test_repeated_requests_stay_correct(self):
        runner = self.make_runner()
        gen = torch.Generator(device="cpu").manual_seed(7)
        with torch.no_grad():
            for _ in range(6):
                x = torch.randn(72, 64, generator=gen).to(NPU_DEV)
                y = torch.randn(72, 64, generator=gen).to(NPU_DEV)
                result = self.run_split(runner, x, y)
                self.expect_close(result, region(self.model, x, y))


if __name__ == "__main__":
    unittest.main()
