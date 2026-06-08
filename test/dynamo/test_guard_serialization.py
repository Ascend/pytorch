# lintrunner: skip PYFMT
# Owner(s): ["module: dynamo"]
"""Module for dynamo guard serialization tests."""

import dataclasses
import sys
import types

import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
import torch._dynamo.package
import torch._inductor.test_case
from torch._dynamo.bytecode_transformation import transform_code_object
from torch._dynamo.guards import CheckFunctionManager, CompileId
from torch._dynamo.symbolic_convert import (
    ExceptionStack,
    InstructionTranslator,
    SpeculationLog,
)
from torch._dynamo.utils import dynamo_timed, get_metrics_context
from torch._guards import compile_context, CompileContext, tracing


@dataclasses.dataclass
class _FrameState:
    f_locals: dict
    f_globals: dict
    f_code: types.CodeType
    f_builtins: dict


class TestGuardSerializationBase(torch._inductor.test_case.TestCase):
    def _tracefunc(self, frame, event, arg):
        if event != "call":
            return
        if self._frame_state is not None:
            return

        self._frame_state = _FrameState(
            f_locals=dict(frame.f_locals),
            f_globals=frame.f_globals,
            f_code=frame.f_code,
            f_builtins=frame.f_builtins,
        )

    def _test_serialization(self, guard_type, fn, *args):
        torch._dynamo.reset()
        self._frame_state = None
        sys.settrace(self._tracefunc)
        try:
            fn(*args)
        finally:
            sys.settrace(None)

        self.assertEqual(self._frame_state is not None, True)

        def guard_filter_fn(guards):
            ret = [
                g.guard_type == guard_type or guard_type in g.derived_guard_types
                for g in guards
            ]
            self.assertEqual(any(ret), True)
            return ret

        ref_gm = None
        loaded_gm = None

        def transform(instructions: list, code_options: dict[str, object]):
            nonlocal ref_gm
            nonlocal loaded_gm

            torch._dynamo.convert_frame.initial_global_state = (
                torch._C._dynamo.guards.GlobalStateGuard()
            )
            tracer = InstructionTranslator(
                instructions,
                self._frame_state.f_code,
                self._frame_state.f_locals,
                self._frame_state.f_globals,
                self._frame_state.f_builtins,
                fn.__closure__ or (),
                torch.overrides._get_current_function_mode_stack(),
                code_options,
                torch._dynamo.lookup_backend("eager"),
                one_graph=False,
                export=False,
                export_constraints=None,
                frame_state=None,
                speculation_log=SpeculationLog(),
                exn_vt_stack=ExceptionStack(),
                distributed_state=None,
                package=None,
            )
            with (
                compile_context(
                    CompileContext(CompileId(frame_id=0, frame_compile_id=0))
                ),
                tracing(tracer.output.tracing_context),
                tracer.set_current_tx(),
                get_metrics_context(),
                dynamo_timed(""),
            ):
                tracer.run()

                ref_gm = CheckFunctionManager(
                    self._frame_state.f_code,
                    tracer.output,
                    guard_filter_fn=guard_filter_fn,
                ).guard_manager

                check_fn_manager = CheckFunctionManager(
                    self._frame_state.f_code,
                    tracer.output,
                    guard_filter_fn=guard_filter_fn,
                    save_guards=True,
                )
                guards_state = check_fn_manager.guards_state
                self.assertEqual(guards_state is not None, True)
                guards_state = torch._dynamo.package.load_guards_state(guards_state)

                loaded_gm = torch._dynamo.package.load_guard_manager(
                    guards_state,
                    self._frame_state.f_code,
                    self._frame_state.f_globals,
                )

        try:
            transform_code_object(self._frame_state.f_code, transform)
        finally:
            torch._dynamo.convert_frame.initial_global_state = None
            self._frame_state = None

        self.assertEqual(ref_gm is not None, True)
        self.assertEqual(loaded_gm is not None, True)
        return ref_gm, loaded_gm

    def _test_check_fn(self, ref, loaded, inputs, expected):
        self.assertEqual(isinstance(inputs, dict), True)
        self.assertEqual(ref.check(inputs), expected)
        self.assertEqual(ref.check(inputs), loaded.check(inputs))


@torch._dynamo.config.patch({"strict_precompile": True})
class TestCachingPrecompileGuardSerialization(TestGuardSerializationBase):
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_id_match_with_config(self):
        def fn(x):
            return x + id(x)

        ref, loaded = self._test_serialization("ID_MATCH", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, True)

        def fn(x):
            # torch.no_grad() installs a CLASS_MATCH guard.
            with torch.no_grad():
                y = x * 2
            return y

        ref, loaded = self._test_serialization("CLASS_MATCH", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, True)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
