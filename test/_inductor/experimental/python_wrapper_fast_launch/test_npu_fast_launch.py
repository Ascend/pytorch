import importlib
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)
PACKAGE = "torch_npu._inductor.experimental.python_wrapper_fast_launch"


@contextmanager
def isolated_fast_launch(c_extension=None, *, fast_launch=True, **config_values):
    for name in list(sys.modules):
        if name == PACKAGE or name.startswith(PACKAGE + "."):
            sys.modules.pop(name, None)

    torch = types.ModuleType("torch")
    torch.__path__ = []
    autograd = types.ModuleType("torch.autograd")
    profiler = types.ModuleType("torch.autograd.profiler")
    profiler._is_profiler_enabled = False
    autograd.profiler = profiler
    torch.autograd = autograd

    torch_npu = types.ModuleType("torch_npu")
    torch_npu.__path__ = [os.path.join(REPO_ROOT, "torch_npu")]
    inductor = types.ModuleType("torch_npu._inductor")
    inductor.__path__ = [os.path.join(REPO_ROOT, "torch_npu", "_inductor")]
    config = types.ModuleType("torch_npu._inductor.config")
    defaults = {
        "dump_fx_graph": False,
        "check_accuracy": False,
        "enable_fast_launch": fast_launch,
    }
    defaults.update(config_values)
    for name, value in defaults.items():
        setattr(config, name, value)
    inductor.config = config

    modules = {
        "torch": torch,
        "torch.autograd": autograd,
        "torch.autograd.profiler": profiler,
        "torch_npu": torch_npu,
        "torch_npu._inductor": inductor,
        "torch_npu._inductor.config": config,
    }
    if c_extension is not None:
        modules["torch_npu._C"] = c_extension
        torch_npu._C = c_extension

    with mock.patch.dict(sys.modules, modules):
        try:
            yield profiler
        finally:
            for name in list(sys.modules):
                if name == PACKAGE or name.startswith(PACKAGE + "."):
                    sys.modules.pop(name, None)


class FakeLauncher:
    def __init__(self, *, arg_kinds=("tensor", "i32")):
        self.config = types.SimpleNamespace(found_by_coordesc=True)
        self.store_cubin = False
        self._npu_fast_launch_kernel_name = "triton_poi_fused_0"
        self._npu_fast_launch_kernel_stub = 1234
        self._npu_fast_launch_kernel_stub_owner = object()
        self._npu_fast_launch_arg_kinds = arg_kinds
        self._npu_fast_launch_grid_exprs = ("xnumel", "1", "1")
        self._npu_fast_launch_get_grid = lambda *args: (2, 1, 1)
        self._npu_fast_launch_enable_simt = False
        self._npu_fast_launch_shared_mem_dynamic_size = 0
        self._npu_fast_launch_is_pure_simt = False
        self._npu_fast_launch_target_support_ffts = False
        self._npu_fast_launch_workspace_size = -1
        self._npu_fast_launch_lock_num = -1
        self._npu_fast_launch_device_print_enabled = False
        self._npu_fast_launch_enter_hook = None
        self._npu_fast_launch_exit_hook = None
        self._npu_fast_launch_enter_hook_callbacks = ()
        self._npu_fast_launch_exit_hook_callbacks = ()
        self.calls = []

    def __call__(self, *args, stream):
        metadata = {"args": args, "stream": stream}
        if self._npu_fast_launch_enter_hook is not None:
            self._npu_fast_launch_enter_hook(metadata)
        self.calls.append((args, stream))
        if self._npu_fast_launch_exit_hook is not None:
            self._npu_fast_launch_exit_hook(metadata)
        return "launcher"


class HookChain:
    def __init__(self, *, reversed=False):
        self.calls = []
        self.reversed = reversed

    def add(self, callback):
        if callback not in self.calls:
            self.calls.append(callback)

    def remove(self, callback):
        if callback in self.calls:
            self.calls.remove(callback)

    def __call__(self, *args, **kwargs):
        callbacks = reversed(self.calls) if self.reversed else self.calls
        for callback in callbacks:
            callback(*args, **kwargs)


def set_launcher_hook_chains(launcher):
    launcher._npu_fast_launch_enter_hook = HookChain()
    launcher._npu_fast_launch_exit_hook = HookChain(reversed=True)
    launcher._npu_fast_launch_enter_hook_callbacks = (
        launcher._npu_fast_launch_enter_hook.calls
    )
    launcher._npu_fast_launch_exit_hook_callbacks = (
        launcher._npu_fast_launch_exit_hook.calls
    )


class FakeTensor:
    def __init__(self, pointer=1):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class FakeAutotuner:
    def __init__(self, launcher):
        self.launchers = [launcher]
        self.best_launcher = None
        self.best_runtime_blocks = ()
        self.inductor_meta = {}
        self.triton_interpret = False
        self.dump_launch_params = False
        self.run_calls = []

    def _build_runtime_launch_args(self, args, runtime_blocks):
        return (*args, *runtime_blocks)

    def run(self, *args, stream, benchmark_run=False, **kwargs):
        self.run_calls.append((args, stream, benchmark_run, kwargs))
        self.best_launcher = self.launchers[0]
        return "fallback"


def metadata(
    *,
    eligible=True,
    schema_state=None,
    arg_kinds=("tensor", "i32"),
    runtime_arg_count=2,
):
    result = {
        "graph_id": "graph-0",
        "callsite_id": "graph-0:0",
        "kernel_name": "triton_poi_fused_0",
        "schema_hash": "schema-0",
        "arg_kinds": arg_kinds,
        "runtime_arg_count": runtime_arg_count,
        "eligible": eligible,
        "fallback_reason": None if eligible else "unsupported",
    }
    if schema_state is not None:
        result["schema_state"] = schema_state
    return result


class TestNPUFastLaunch(unittest.TestCase):
    def test_disabled_environment_forces_cached_wrapper_to_full_entry(self):
        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=mock.Mock(),
            _npu_inductor_fast_launch_with_plan=mock.Mock(),
        )
        with isolated_fast_launch(extension, fast_launch=False):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            launcher = FakeLauncher()
            autotuner = FakeAutotuner(launcher)
            autotuner.best_launcher = launcher
            bound = bind.BoundFastLaunch(autotuner, metadata())
            tensor = FakeTensor()

            self.assertEqual(bound(tensor, 3, stream=99), "fallback")
            self.assertEqual(bound(tensor, 4, stream=99), "fallback")

        self.assertEqual(len(autotuner.run_calls), 2)
        self.assertEqual(launcher.calls, [])
        extension._npu_inductor_make_fast_launch_plan.assert_not_called()
        extension._npu_inductor_fast_launch_with_plan.assert_not_called()

    def test_codegen_missing_signature_is_promotable_incomplete_schema(self):
        with isolated_fast_launch():
            codegen = importlib.import_module(f"{PACKAGE}.codegen")
            result = codegen.build_callsite_metadata(
                kernel_name="triton_poi_fused_0",
                call_args=("buf0", "xnumel"),
                triton_meta={},
                graph_id="graph-0",
                callsite_index=0,
            )

        self.assertEqual(result["schema_state"], "incomplete")
        self.assertEqual(result["schema_reason"], "codegen_signature_missing")
        self.assertEqual(result["runtime_arg_count"], 2)
        self.assertTrue(result["eligible"])

    def test_codegen_complete_schema_keeps_ordered_arg_kinds(self):
        with isolated_fast_launch():
            codegen = importlib.import_module(f"{PACKAGE}.codegen")
            result = codegen.build_callsite_metadata(
                kernel_name="triton_poi_fused_0",
                call_args=("buf0", "xnumel"),
                triton_meta={
                    "signature": {"in_ptr0": "*fp32", "xnumel": "i32"},
                    "constants": {},
                },
                graph_id="graph-0",
                callsite_index=0,
            )

        self.assertEqual(result["schema_state"], "complete")
        self.assertEqual(result["arg_kinds"], ("tensor", "i32"))

    def test_plan_owns_binary_and_launches_with_tuple_args(self):
        calls = []

        class Plan:
            pass

        def make_plan(*args):
            calls.append(("make", args))
            return Plan()

        def launch(*args):
            calls.append(("launch", args))

        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=make_plan,
            _npu_inductor_fast_launch_with_plan=launch,
        )
        with isolated_fast_launch(extension):
            backend = importlib.import_module(f"{PACKAGE}.backend")
            launcher = FakeLauncher()
            tensor = FakeTensor()
            planned = backend.build_planned_fast_launch(
                launcher,
                metadata(),
                canonical_args=(tensor, 3),
                runtime_arg_count=2,
            )
            planned((tensor, 3), stream=99)

        self.assertIs(planned.plan._owner, launcher._npu_fast_launch_kernel_stub_owner)
        self.assertEqual(calls[1][0], "launch")
        self.assertEqual(calls[1][1][1:5], (99, 2, 1, 1))
        self.assertEqual(calls[1][1][5], (tensor, 3))
        self.assertFalse(calls[0][1][-1])

    def test_plan_forwards_ffts_abi_requirement(self):
        calls = []

        class Plan:
            pass

        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=lambda *args: (
                calls.append(args) or Plan()
            ),
            _npu_inductor_fast_launch_with_plan=lambda *args: None,
        )
        with isolated_fast_launch(extension):
            backend = importlib.import_module(f"{PACKAGE}.backend")
            launcher = FakeLauncher()
            launcher._npu_fast_launch_target_support_ffts = True
            backend.build_planned_fast_launch(
                launcher,
                metadata(),
                canonical_args=(FakeTensor(), 3),
                runtime_arg_count=2,
            )

        self.assertTrue(calls[0][-1])

    def test_launcher_requiring_hidden_resources_is_negative(self):
        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=mock.Mock(),
            _npu_inductor_fast_launch_with_plan=mock.Mock(),
        )
        with isolated_fast_launch(extension):
            backend = importlib.import_module(f"{PACKAGE}.backend")
            types_module = importlib.import_module(f"{PACKAGE}.types")
            for attribute, reason in (
                ("_npu_fast_launch_workspace_size", "launcher_workspace_required"),
                ("_npu_fast_launch_lock_num", "launcher_sync_block_lock_required"),
            ):
                launcher = FakeLauncher()
                setattr(launcher, attribute, 1)
                with self.subTest(attribute=attribute), self.assertRaises(
                    types_module.FastLaunchPlanUnavailable
                ) as error:
                    backend.build_planned_fast_launch(
                        launcher,
                        metadata(),
                        canonical_args=(FakeTensor(), 3),
                        runtime_arg_count=2,
                    )
                self.assertEqual(str(error.exception), reason)

        extension._npu_inductor_make_fast_launch_plan.assert_not_called()

    def test_warmup_then_promotes_to_planned_backend(self):
        launches = []

        class Plan:
            pass

        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=lambda *args: Plan(),
            _npu_inductor_fast_launch_with_plan=lambda *args: launches.append(args),
        )
        with isolated_fast_launch(extension):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            launcher = FakeLauncher()
            autotuner = FakeAutotuner(launcher)
            bound = bind.BoundFastLaunch(autotuner, metadata())
            tensor = FakeTensor()
            self.assertEqual(bound(tensor, 3, stream=99), "fallback")
            self.assertIsNone(bound(tensor, 4, stream=99))

        self.assertEqual(len(autotuner.run_calls), 1)
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0][5], (tensor, 4))

    def test_cold_coordinate_descent_promotes_after_autotune(self):
        launches = []

        class Plan:
            pass

        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=lambda *args: Plan(),
            _npu_inductor_fast_launch_with_plan=lambda *args: launches.append(args),
        )
        with isolated_fast_launch(extension):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            launcher = FakeLauncher()
            launcher.config.found_by_coordesc = False
            autotuner = FakeAutotuner(launcher)
            autotuner.inductor_meta["coordinate_descent_tuning"] = True
            bound = bind.BoundFastLaunch(autotuner, metadata())
            tensor = FakeTensor()
            self.assertEqual(bound(tensor, 3, stream=99), "fallback")
            self.assertIsNone(bound(tensor, 4, stream=99))

        self.assertEqual(len(autotuner.run_calls), 1)
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0][5], (tensor, 4))

    def test_grouped_autotuner_keeps_original_entry(self):
        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=mock.Mock(),
            _npu_inductor_fast_launch_with_plan=mock.Mock(),
        )
        with isolated_fast_launch(extension):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            launcher = FakeLauncher()
            autotuner = FakeAutotuner(launcher)
            autotuner.inductor_meta["group_enabled"] = True
            autotuner.best_launcher_map = {}
            call_slot = [None]
            call = bind.bind_python_wrapper_kernel_fast(
                metadata(),
                autotuner,
                call_slot=call_slot,
            )
            tensor = FakeTensor()
            self.assertIs(call_slot[0], call)
            self.assertNotIsInstance(call, bind.BoundFastLaunch)
            self.assertEqual(call(tensor, 3, stream=99), "fallback")
            self.assertEqual(call(tensor, 4, stream=99), "fallback")

        self.assertEqual(len(autotuner.run_calls), 2)
        extension._npu_inductor_make_fast_launch_plan.assert_not_called()
        extension._npu_inductor_fast_launch_with_plan.assert_not_called()

    def test_negative_cache_calls_stable_original_launcher(self):
        extension = types.SimpleNamespace()
        with isolated_fast_launch(extension):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            launcher = FakeLauncher()
            autotuner = FakeAutotuner(launcher)
            bound = bind.BoundFastLaunch(autotuner, metadata(eligible=False))
            tensor = FakeTensor()
            self.assertEqual(bound(tensor, 3, stream=99), "fallback")
            self.assertEqual(bound(tensor, 4, stream=99), "launcher")

        self.assertEqual(len(autotuner.run_calls), 1)
        self.assertEqual(launcher.calls, [((tensor, 4), 99)])

    def test_profiler_forces_full_entry(self):
        extension = types.SimpleNamespace()
        with isolated_fast_launch(extension) as profiler:
            bind = importlib.import_module(f"{PACKAGE}.bind")
            launcher = FakeLauncher()
            autotuner = FakeAutotuner(launcher)
            autotuner.best_launcher = launcher
            profiler._is_profiler_enabled = True
            bound = bind.BoundFastLaunch(autotuner, metadata())
            self.assertEqual(bound(FakeTensor(), 3, stream=99), "fallback")

        self.assertEqual(len(autotuner.run_calls), 1)

    def test_launch_hook_callback_source_tracks_hook_chain_mutation(self):
        with isolated_fast_launch():
            launcher_module = importlib.import_module(f"{PACKAGE}.launcher")
            chain = HookChain()

            def callback(metadata):
                return None

            callbacks = launcher_module._launch_hook_callbacks(chain)
            self.assertIs(callbacks, chain.calls)
            self.assertFalse(callbacks)

            chain.add(callback)
            self.assertTrue(callbacks)
            chain.remove(callback)
            self.assertFalse(callbacks)

            def legacy_hook(metadata):
                return None

            self.assertEqual(
                launcher_module._launch_hook_callbacks(legacy_hook),
                (legacy_hook,),
            )

    def test_empty_triton_launch_hook_chain_does_not_block_fast_path(self):
        launches = []

        class Plan:
            pass

        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=lambda *args: Plan(),
            _npu_inductor_fast_launch_with_plan=lambda *args: launches.append(args),
        )
        with isolated_fast_launch(extension):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            launcher = FakeLauncher()
            set_launcher_hook_chains(launcher)
            autotuner = FakeAutotuner(launcher)
            bound = bind.BoundFastLaunch(autotuner, metadata())
            tensor = FakeTensor()
            self.assertEqual(bound(tensor, 3, stream=99), "fallback")
            self.assertIsNone(bound(tensor, 4, stream=99))

        self.assertEqual(len(autotuner.run_calls), 1)
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0][5], (tensor, 4))

    def test_active_triton_launch_hooks_use_original_launcher(self):
        launches = []
        hook_events = []

        class Plan:
            pass

        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=lambda *args: Plan(),
            _npu_inductor_fast_launch_with_plan=lambda *args: launches.append(args),
        )
        with isolated_fast_launch(extension):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            launcher = FakeLauncher()
            set_launcher_hook_chains(launcher)
            launcher._npu_fast_launch_enter_hook.add(
                lambda metadata: hook_events.append(("enter", metadata))
            )
            launcher._npu_fast_launch_exit_hook.add(
                lambda metadata: hook_events.append(("exit", metadata))
            )
            autotuner = FakeAutotuner(launcher)
            bound = bind.BoundFastLaunch(autotuner, metadata())
            tensor = FakeTensor()
            self.assertEqual(bound(tensor, 3, stream=99), "fallback")
            self.assertEqual(bound(tensor, 4, stream=100), "launcher")

        self.assertEqual(len(autotuner.run_calls), 1)
        self.assertEqual(launches, [])
        self.assertEqual(launcher.calls, [((tensor, 4), 100)])
        self.assertEqual([event[0] for event in hook_events], ["enter", "exit"])
        self.assertEqual(hook_events[0][1]["stream"], 100)

    def test_triton_launch_hook_activation_temporarily_falls_back(self):
        launches = []
        hook_events = []

        class Plan:
            pass

        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=lambda *args: Plan(),
            _npu_inductor_fast_launch_with_plan=lambda *args: launches.append(args),
        )
        with isolated_fast_launch(extension):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            launcher = FakeLauncher()
            set_launcher_hook_chains(launcher)
            autotuner = FakeAutotuner(launcher)
            bound = bind.BoundFastLaunch(autotuner, metadata())
            tensor = FakeTensor()

            self.assertEqual(bound(tensor, 3, stream=99), "fallback")
            self.assertIsNone(bound(tensor, 4, stream=100))

            def enter(metadata):
                hook_events.append(("enter", metadata))

            def exit(metadata):
                hook_events.append(("exit", metadata))

            launcher._npu_fast_launch_enter_hook.add(enter)
            launcher._npu_fast_launch_exit_hook.add(exit)
            self.assertEqual(bound(tensor, 5, stream=101), "launcher")

            launcher._npu_fast_launch_enter_hook.remove(enter)
            launcher._npu_fast_launch_exit_hook.remove(exit)
            self.assertIsNone(bound(tensor, 6, stream=102))

        self.assertEqual(len(autotuner.run_calls), 1)
        self.assertEqual(
            [launch[5] for launch in launches],
            [(tensor, 4), (tensor, 6)],
        )
        self.assertEqual(launcher.calls, [((tensor, 5), 101)])
        self.assertEqual([event[0] for event in hook_events], ["enter", "exit"])

    def test_dynamic_grid_failure_does_not_poison_plan(self):
        launches = []

        class Plan:
            pass

        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=lambda *args: Plan(),
            _npu_inductor_fast_launch_with_plan=lambda *args: launches.append(args),
        )
        with isolated_fast_launch(extension):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            launcher = FakeLauncher()
            launcher._npu_fast_launch_get_grid = lambda tensor, size: (size, 1, 1)
            autotuner = FakeAutotuner(launcher)
            bound = bind.BoundFastLaunch(autotuner, metadata())
            tensor = FakeTensor()
            self.assertEqual(bound(tensor, 2, stream=99), "fallback")
            self.assertEqual(bound(tensor, 70_000, stream=99), "fallback")
            self.assertIsNone(bound(tensor, 3, stream=100))

        self.assertEqual(len(autotuner.run_calls), 2)
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0][1:5], (100, 3, 1, 1))

    def test_backend_error_never_replays_fallback(self):
        class Plan:
            pass

        def fail_after_boundary(*args):
            raise RuntimeError("rtKernelLaunch failed")

        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=lambda *args: Plan(),
            _npu_inductor_fast_launch_with_plan=fail_after_boundary,
        )
        with isolated_fast_launch(extension):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            types_module = importlib.import_module(f"{PACKAGE}.types")
            launcher = FakeLauncher()
            autotuner = FakeAutotuner(launcher)
            bound = bind.BoundFastLaunch(autotuner, metadata())
            tensor = FakeTensor()
            self.assertEqual(bound(tensor, 3, stream=99), "fallback")
            with self.assertRaises(types_module.FastLaunchError) as error:
                bound(tensor, 4, stream=99)
            self.assertTrue(error.exception.backend_submitted)

        self.assertEqual(len(autotuner.run_calls), 1)

    def test_incomplete_schema_is_completed_from_launcher_abi(self):
        launches = []

        class Plan:
            pass

        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=lambda *args: Plan(),
            _npu_inductor_fast_launch_with_plan=lambda *args: launches.append(args),
        )
        with isolated_fast_launch(extension):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            launcher = FakeLauncher(
                arg_kinds=("tensor", "tensor", "i32", "bool", "i32")
            )
            autotuner = FakeAutotuner(launcher)
            autotuner.best_runtime_blocks = (16,)
            bound = bind.BoundFastLaunch(
                autotuner,
                metadata(
                    schema_state="incomplete",
                    arg_kinds=(),
                    runtime_arg_count=4,
                ),
            )
            first = FakeTensor(1)
            second = FakeTensor(2)
            self.assertEqual(bound(first, second, 7, True, stream=99), "fallback")
            self.assertIsNone(bound(first, second, 8, False, stream=100))

        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0][5], (first, second, 8, False, 16))

    def test_complete_codegen_launcher_schema_conflict_is_negative(self):
        launches = []

        class Plan:
            pass

        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=lambda *args: Plan(),
            _npu_inductor_fast_launch_with_plan=lambda *args: launches.append(args),
        )
        with isolated_fast_launch(extension):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            launcher = FakeLauncher(arg_kinds=("tensor", "i64"))
            autotuner = FakeAutotuner(launcher)
            bound = bind.BoundFastLaunch(autotuner, metadata())
            tensor = FakeTensor()
            self.assertEqual(bound(tensor, 3, stream=99), "fallback")
            self.assertEqual(bound(tensor, 4, stream=99), "launcher")

        self.assertEqual(launches, [])
        self.assertEqual(len(autotuner.run_calls), 1)

    def test_negative_hit_skips_launcher_stability_recheck(self):
        extension = types.SimpleNamespace()
        with isolated_fast_launch(extension):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            launcher = FakeLauncher()
            autotuner = FakeAutotuner(launcher)
            bound = bind.BoundFastLaunch(autotuner, metadata(eligible=False))
            tensor = FakeTensor()
            self.assertEqual(bound(tensor, 3, stream=99), "fallback")
            with mock.patch.object(
                bind.BoundFastLaunch,
                "_stable_launcher",
                side_effect=AssertionError("stable launcher was rechecked"),
            ):
                self.assertEqual(bound(tensor, 4, stream=99), "launcher")

    def test_negative_cache_is_invalidated_when_launcher_changes(self):
        extension = types.SimpleNamespace()
        with isolated_fast_launch(extension):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            first_launcher = FakeLauncher()
            second_launcher = FakeLauncher()
            autotuner = FakeAutotuner(first_launcher)
            bound = bind.BoundFastLaunch(autotuner, metadata(eligible=False))
            tensor = FakeTensor()
            self.assertEqual(bound(tensor, 3, stream=99), "fallback")
            autotuner.launchers = [second_launcher]
            autotuner.best_launcher = second_launcher
            self.assertEqual(bound(tensor, 4, stream=99), "fallback")

        self.assertEqual(first_launcher.calls, [])
        self.assertEqual(len(autotuner.run_calls), 2)

    def test_no_runtime_blocks_reuses_args_without_builder_call(self):
        launches = []

        class Plan:
            pass

        extension = types.SimpleNamespace(
            _npu_inductor_make_fast_launch_plan=lambda *args: Plan(),
            _npu_inductor_fast_launch_with_plan=lambda *args: launches.append(args),
        )
        with isolated_fast_launch(extension):
            bind = importlib.import_module(f"{PACKAGE}.bind")
            launcher = FakeLauncher()
            autotuner = FakeAutotuner(launcher)
            autotuner._build_runtime_launch_args = mock.Mock(
                side_effect=AssertionError("runtime arg builder should be cold")
            )
            bound = bind.BoundFastLaunch(autotuner, metadata())
            tensor = FakeTensor()
            self.assertEqual(bound(tensor, 3, stream=99), "fallback")
            self.assertIsNone(bound(tensor, 4, stream=99))

        self.assertEqual(len(launches), 1)
        autotuner._build_runtime_launch_args.assert_not_called()


if __name__ == "__main__":
    unittest.main()
