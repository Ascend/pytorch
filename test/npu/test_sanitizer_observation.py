# Copyright (c) 2026, Huawei Technologies Co., Ltd
# Owner(s): ["module: npu"]
"""Tests for the NPU sanitizer observation layers and mount lifecycle
(torch_npu/npu/_sanitizer.py). Collectives are fakes here — the
wrappers only observe arguments, so no process group is needed."""

import gc
import os
import tempfile
import warnings
from unittest import mock

import torch
import torch.distributed as dist
import torch_npu
from torch.autograd import Variable
from torch_npu.testing.testcase import TestCase, run_tests


def setup_sanitizer():
    """Enable sanitizer with record_stream checking."""
    os.environ['TORCH_NPU_SANITIZER'] = '1'
    import torch_npu.npu._sanitizer as sanitizer
    if not sanitizer.npu_sanitizer.enabled:
        sanitizer.npu_sanitizer.enable()


def reset_sanitizer():
    """Reset sanitizer state between tests."""
    import torch_npu.npu._sanitizer as sanitizer
    if sanitizer.npu_sanitizer.dispatch is not None:
        try:
            sanitizer.npu_sanitizer.dispatch.__exit__(None, None, None)
        except Exception:
            pass
        sanitizer.npu_sanitizer.dispatch = None
    sanitizer.npu_sanitizer.event_handler = None
    sanitizer.npu_sanitizer.enabled = False


def get_event_handler():
    import torch_npu.npu._sanitizer as sanitizer
    return sanitizer.npu_sanitizer.event_handler


def get_sanitizer_module():
    import torch_npu.npu._sanitizer as sanitizer
    return sanitizer


def get_engine_observation():
    return get_sanitizer_module()._engine_observation


def get_dist_observation():
    return get_sanitizer_module()._dist_observation


def make_backward_graph():
    """A leaf/root pair allocated on the current (default) stream."""
    x = torch.randn(100, device="npu", requires_grad=True)
    y = x * 2
    grad = torch.ones(100, device="npu")
    return x, y, grad


def _fake_comm(*args, **kwargs):
    return None


class SanitizerObservationTestBase(TestCase):
    def setUp(self):
        reset_sanitizer()
        setup_sanitizer()
        get_engine_observation().extraction_warned = False

    def tearDown(self):
        reset_sanitizer()
        get_engine_observation().uninstall()
        get_dist_observation().uninstall()

    @staticmethod
    def _tracked_info(tensor):
        handler = get_event_handler()
        ptr = tensor.untyped_storage().data_ptr()
        info = handler._npu_tensors.get(ptr)
        if info is None:
            raise AssertionError(
                f"Tensor storage {ptr} is not tracked by NPU sanitizer."
            )
        return handler, ptr, info


class SanitizerDistTestBase(TestCase):
    def setUp(self):
        reset_sanitizer()
        self._saved = {
            name: getattr(dist, name, None)
            for name in get_dist_observation().SPECS
        }
        for name, orig in self._saved.items():
            if orig is not None:
                setattr(dist, name, _fake_comm)
        setup_sanitizer()
        get_dist_observation().warned.clear()

    def tearDown(self):
        get_dist_observation().uninstall()
        for name, orig in self._saved.items():
            if orig is not None:
                setattr(dist, name, orig)
        reset_sanitizer()
        get_engine_observation().uninstall()

    @staticmethod
    def _tracked_info(tensor):
        handler = get_event_handler()
        ptr = tensor.untyped_storage().data_ptr()
        info = handler._npu_tensors.get(ptr)
        if info is None:
            raise AssertionError(
                f"Tensor storage {ptr} is not tracked by NPU sanitizer."
            )
        return handler, ptr, info

    @staticmethod
    def _side_stream_context():
        side = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()
        side.wait_stream(default_stream)
        return side, default_stream


class TestEngineInstallLifecycle(SanitizerObservationTestBase):
    def test_enable_installs_proxy(self):
        engine_check = get_engine_observation()
        proxy = Variable._execution_engine
        self.assertIsInstance(proxy, get_sanitizer_module()._EngineProxy)
        self.assertIs(proxy.handler, get_event_handler())

    def test_reenable_rebinds_handler_without_rewrapping(self):
        engine_check = get_engine_observation()
        proxy_before = Variable._execution_engine
        reset_sanitizer()
        setup_sanitizer()
        proxy_after = Variable._execution_engine
        self.assertIs(proxy_before, proxy_after)
        self.assertIs(proxy_after.handler, get_event_handler())
        self.assertIs(proxy_after._orig, engine_check.orig_engine)

    def test_uninstall_restores_original(self):
        engine_check = get_engine_observation()
        orig = engine_check.orig_engine
        proxy = Variable._execution_engine
        engine_check.uninstall()
        self.assertIs(Variable._execution_engine, orig)
        self.assertIsNone(proxy.handler)
        reset_sanitizer()
        setup_sanitizer()
        self.assertIsInstance(Variable._execution_engine, get_sanitizer_module()._EngineProxy)

    def test_reinstall_reuses_proxy_restored_by_third_party(self):
        """A proxy restored into the engine slot by a third party
        after an uninstall is reused on reinstall, not rewrapped."""
        engine_check = get_engine_observation()
        proxy = Variable._execution_engine

        third_party_engine = object()

        Variable._execution_engine = third_party_engine
        engine_check.uninstall()
        Variable._execution_engine = proxy

        engine_check.install(get_event_handler())
        self.assertIs(Variable._execution_engine, proxy)
        self.assertIs(engine_check.proxy, proxy)
        self.assertIs(proxy.handler, get_event_handler())


class TestEngineRecording(SanitizerObservationTestBase):
    def test_entry_records_roots_and_grads(self):
        x, y, grad = make_backward_graph()
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()
        stream.wait_stream(default_stream)

        with torch_npu.npu.stream(stream):
            torch.autograd.backward(y, grad)

        _, y_ptr, y_info = self._tracked_info(y)
        _, _, grad_info = self._tracked_info(grad)
        engine_stream = int(stream.npu_stream)
        self.assertIn(engine_stream, y_info.cross_stream_usages)
        self.assertIn(engine_stream, grad_info.cross_stream_usages)
        self.assertEqual(
            y_info.cross_stream_usages[engine_stream].operator,
            "autograd_engine [engine]",
        )

    def test_exit_fallback_records_leaf_grad(self):
        x = torch.randn(100, device="npu", requires_grad=True)
        preexisting_leaf_grad = torch.zeros(100, device="npu")
        x.grad = preexisting_leaf_grad
        grad = torch.ones(100, device="npu")
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()
        stream.wait_stream(default_stream)

        with torch_npu.npu.stream(stream):
            torch.autograd.backward(x, grad)

        _, leaf_ptr, leaf_info = self._tracked_info(preexisting_leaf_grad)
        engine_stream = int(stream.npu_stream)
        self.assertIn(engine_stream, leaf_info.cross_stream_usages)
        self.assertEqual(x.grad.untyped_storage().data_ptr(), leaf_ptr)

    def test_inputs_empty_non_leaf_exit_not_recorded(self):
        """``inputs=()`` non-leaf subgraph: no return channel, so the
        exit side records nothing (documented limitation)."""
        x = torch.randn(100, device="npu", requires_grad=True)
        preexisting_leaf_grad = torch.zeros(100, device="npu")
        x.grad = preexisting_leaf_grad
        y = x * 2
        grad = torch.ones(100, device="npu")
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()
        stream.wait_stream(default_stream)

        with torch_npu.npu.stream(stream):
            Variable._execution_engine.run_backward(
                tensors=(y,),
                grad_tensors=(grad,),
                keep_graph=False,
                create_graph=False,
                inputs=(),
                allow_unreachable=True,
                accumulate_grad=True,
            )

        engine_stream = int(stream.npu_stream)
        _, _, y_info = self._tracked_info(y)
        _, _, grad_info = self._tracked_info(grad)
        self.assertIn(engine_stream, y_info.cross_stream_usages)
        self.assertIn(engine_stream, grad_info.cross_stream_usages)
        _, _, leaf_info = self._tracked_info(preexisting_leaf_grad)
        self.assertNotIn(engine_stream, leaf_info.cross_stream_usages)

    def test_same_stream_backward_not_recorded(self):
        x, y, grad = make_backward_graph()
        torch.autograd.backward(y, grad)
        _, _, y_info = self._tracked_info(y)
        self.assertEqual(len(y_info.cross_stream_usages), 0)
        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertEqual(len(warnings), 0)

    def test_kwargs_form_engine_call_recorded(self):
        """Direct engine call with keyword-only arguments (the shape used
        by graph runners such as MindSpeed's ``run_graph_backward``) goes
        through the same entry extraction and exit leaf-grad fallback."""
        x = torch.randn(100, device="npu", requires_grad=True)
        preexisting_leaf_grad = torch.zeros(100, device="npu")
        x.grad = preexisting_leaf_grad
        grad = torch.ones(100, device="npu")
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()
        stream.wait_stream(default_stream)

        with torch_npu.npu.stream(stream):
            Variable._execution_engine.run_backward(
                tensors=(x,),
                grad_tensors=(grad,),
                keep_graph=False,
                create_graph=False,
                inputs=(),
                allow_unreachable=True,
                accumulate_grad=True,
            )

        engine_stream = int(stream.npu_stream)
        _, _, x_info = self._tracked_info(x)
        _, _, grad_info = self._tracked_info(grad)
        _, _, leaf_info = self._tracked_info(preexisting_leaf_grad)
        self.assertIn(engine_stream, x_info.cross_stream_usages)
        self.assertIn(engine_stream, grad_info.cross_stream_usages)
        self.assertIn(engine_stream, leaf_info.cross_stream_usages)

    def test_flatten_handles_nested_and_none(self):
        sanitizer = get_sanitizer_module()
        t1 = torch.randn(4, device="npu")
        t2 = torch.randn(4, device="npu")
        cpu = torch.randn(4, device="cpu")
        out = set()
        sanitizer._flatten_ptrs([[t1, (t2, None)], cpu], out)
        self.assertEqual(
            out,
            {
                t1.untyped_storage().data_ptr(),
                t2.untyped_storage().data_ptr(),
            },
        )


class TestEngineFreeTimeIntegration(SanitizerObservationTestBase):
    def _cross_stream_grad_setup(self):
        """Cross-stream grad consumed by a backward with the
        consumption-side sync in place — exactly the free-time hazard
        the engine observation must report."""
        x, y, _ = make_backward_graph()
        side = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()
        side.wait_stream(default_stream)
        with torch_npu.npu.stream(side):
            grad = torch.ones(100, device="npu")
        default_stream.wait_stream(side)
        return x, y, grad, side, default_stream

    def test_cross_stream_backward_reported_on_flush(self):
        x, y, grad, side, default_stream = self._cross_stream_grad_setup()
        grad_ptr = grad.untyped_storage().data_ptr()

        torch.autograd.backward(y, grad)

        warnings = get_event_handler().flush_record_stream_warnings()
        grad_warnings = [w for w in warnings if w.data_ptr == grad_ptr]
        self.assertEqual(len(grad_warnings), 1)
        self.assertEqual(
            grad_warnings[0].usage_stream, int(default_stream.npu_stream)
        )

    def test_grad_form_cross_stream_reported(self):
        x, y, grad, side, default_stream = self._cross_stream_grad_setup()
        grad_ptr = grad.untyped_storage().data_ptr()

        torch.autograd.grad(y, x, grad_outputs=grad)

        warnings = get_event_handler().flush_record_stream_warnings()
        grad_warnings = [w for w in warnings if w.data_ptr == grad_ptr]
        self.assertEqual(len(grad_warnings), 1)

    def test_record_stream_suppresses_report(self):
        x, y, grad, side, default_stream = self._cross_stream_grad_setup()
        grad_ptr = grad.untyped_storage().data_ptr()

        grad.record_stream(default_stream)
        torch.autograd.backward(y, grad)

        warnings = get_event_handler().flush_record_stream_warnings()
        grad_warnings = [w for w in warnings if w.data_ptr == grad_ptr]
        self.assertEqual(len(grad_warnings), 0)

    def test_wait_stream_edge_suppresses_report(self):
        x, y, grad, side, default_stream = self._cross_stream_grad_setup()
        grad_ptr = grad.untyped_storage().data_ptr()

        torch.autograd.backward(y, grad)
        side.wait_stream(default_stream)

        warnings = get_event_handler().flush_record_stream_warnings()
        grad_warnings = [w for w in warnings if w.data_ptr == grad_ptr]
        self.assertEqual(len(grad_warnings), 0)

    def test_cross_stream_backward_reported_on_dealloc(self):
        x, y, grad, side, default_stream = self._cross_stream_grad_setup()
        grad_ptr = grad.untyped_storage().data_ptr()

        torch.autograd.backward(y, grad)
        del y, grad, x
        gc.collect()

        errors = get_event_handler().record_stream_errors
        grad_errors = [e for e in errors if e.data_ptr == grad_ptr]
        self.assertGreater(len(grad_errors), 0)

    def test_resize_zero_free_reports_by_design(self):
        """``run_graph_backward`` + ``resize_(0)`` release shape:
        reports by design — safe only under sync-before-reuse, not
        sync-before-free."""
        x, y, grad, side, default_stream = self._cross_stream_grad_setup()
        grad_ptr = grad.untyped_storage().data_ptr()

        torch.autograd.backward(y, grad)
        grad.untyped_storage().resize_(0)
        gc.collect()

        errors = get_event_handler().record_stream_errors
        grad_errors = [e for e in errors if e.data_ptr == grad_ptr]
        self.assertGreater(len(grad_errors), 0)


class TestEngineLifecycleInteractions(SanitizerObservationTestBase):
    def test_protocol_drift_warns_once(self):
        """A drifted engine calling shape raises a one-time degradation
        warning instead of silently recording nothing."""
        engine_check = get_engine_observation()
        self.assertFalse(engine_check.extraction_warned)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(TypeError):
                Variable._execution_engine.run_backward(roots=(object(),))
        self.assertTrue(
            any("protocol may have changed" in str(w.message) for w in caught)
        )
        self.assertTrue(engine_check.extraction_warned)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(TypeError):
                Variable._execution_engine.run_backward(roots=(object(),))
        self.assertFalse(
            any("protocol may have changed" in str(w.message) for w in caught)
        )

    def test_uninstalled_backward_records_nothing(self):
        engine_check = get_engine_observation()
        engine_check.uninstall()
        x, y, grad = make_backward_graph()
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()
        stream.wait_stream(default_stream)

        with torch_npu.npu.stream(stream):
            torch.autograd.backward(y, grad)

        warnings = get_event_handler().flush_record_stream_warnings()
        engine_warnings = [
            w for w in warnings
            if w.usage.operator == "autograd_engine [engine]"
        ]
        self.assertEqual(len(engine_warnings), 0)

    def test_current_stream_failure_neither_crashes_nor_records(self):
        x, y, grad = make_backward_graph()
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()
        stream.wait_stream(default_stream)

        with torch_npu.npu.stream(stream):
            with mock.patch(
                "torch_npu.npu.current_stream", side_effect=RuntimeError
            ):
                torch.autograd.backward(y, grad)

        warnings = get_event_handler().flush_record_stream_warnings()
        engine_warnings = [
            w for w in warnings
            if w.usage.operator == "autograd_engine [engine]"
        ]
        self.assertEqual(len(engine_warnings), 0)

    def test_dispatch_reports_unchanged_with_engine_layer(self):
        x = torch.randn(100, device="npu")
        stream = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        stream.wait_stream(default_stream)
        with torch_npu.npu.stream(stream):
            _ = x + 1

        warnings = get_event_handler().flush_record_stream_warnings()
        dispatch_warnings = [
            w for w in warnings if w.usage_stream == int(stream.npu_stream)
        ]
        self.assertGreater(len(dispatch_warnings), 0)
        self.assertNotEqual(
            dispatch_warnings[0].usage.operator, "autograd_engine [engine]"
        )


class TestDistInstallLifecycle(SanitizerDistTestBase):
    def test_enable_installs_wrappers(self):
        dist_check = get_dist_observation()
        self.assertIs(dist.all_reduce, dist_check.wrappers["all_reduce"])
        self.assertIs(dist.all_reduce.handler, get_event_handler())
        self.assertTrue(dist_check.installed)

    def test_reenable_rebinds_without_rewrapping(self):
        dist_check = get_dist_observation()
        wrapper_before = dist.all_reduce
        reset_sanitizer()
        setup_sanitizer()
        self.assertIs(dist.all_reduce, wrapper_before)
        self.assertIs(dist.all_reduce.handler, get_event_handler())

    def test_uninstall_restores_current_functions(self):
        dist_check = get_dist_observation()
        wrapper = dist.all_reduce
        dist_check.uninstall()
        self.assertIs(dist.all_reduce, _fake_comm)
        self.assertFalse(dist_check.installed)
        self.assertIsNone(wrapper.handler)

    def test_missing_entry_warns_once_at_install(self):
        """A SPECS name absent from the current torch.distributed
        is skipped with a one-time warning at install time (loud
        degradation when a future torch removes or renames an entry)."""
        dist_check = get_dist_observation()
        dist_check.uninstall()
        with mock.patch.dict(
            dist_check.SPECS,
            {"no_such_collective": ((0,), (0,), ("tensor",), ("tensor",))},
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                dist_check.install(get_event_handler())
            self.assertTrue(
                any(
                    "no_such_collective" in str(w.message)
                    for w in caught
                )
            )
            dist_check.uninstall()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                dist_check.install(get_event_handler())
            self.assertFalse(
                any(
                    "no_such_collective" in str(w.message)
                    for w in caught
                )
            )

    def test_install_logs_wrapped_entry_count(self):
        """Install logs how many entry functions are actually wrapped,
        so the observation coverage is confirmable on site. Entries
        missing on this torch are skipped by design, so the wrapped
        count follows the names that actually exist here, not the
        full spec table."""
        dist_check = get_dist_observation()
        expected = sum(
            1 for name in dist_check.SPECS
            if getattr(dist, name, None) is not None
        )
        self.assertGreater(expected, 0)
        with self.assertLogs(
            "torch_npu.npu._sanitizer", level="INFO"
        ) as captured:
            dist_check.install(get_event_handler())
        self.assertTrue(
            any(
                f"observing {expected} torch.distributed"
                in line
                for line in captured.output
            )
        )


class TestDistRecording(SanitizerDistTestBase):
    def test_records_input_and_output_on_current_stream(self):
        inp = torch.randn(64, device="npu")
        out = torch.empty(64, device="npu")
        side, _ = self._side_stream_context()

        with torch_npu.npu.stream(side):
            dist.all_gather_into_tensor(out, inp)

        side_id = int(side.npu_stream)
        _, _, inp_info = self._tracked_info(inp)
        _, _, out_info = self._tracked_info(out)
        self.assertIn(side_id, inp_info.cross_stream_usages)
        self.assertIn(side_id, out_info.cross_stream_usages)
        self.assertEqual(
            out_info.cross_stream_usages[side_id].operator,
            "dist [all_gather_into_tensor]",
        )

    def test_same_stream_usage_not_recorded(self):
        t = torch.randn(64, device="npu")
        dist.all_reduce(t)
        _, _, info = self._tracked_info(t)
        self.assertEqual(len(info.cross_stream_usages), 0)
        reports = get_event_handler().flush_record_stream_warnings()
        self.assertEqual(len(reports), 0)

    def test_kwargs_form_recorded(self):
        out = torch.empty(64, device="npu")
        inp = torch.randn(64, device="npu")
        side, _ = self._side_stream_context()

        with torch_npu.npu.stream(side):
            dist.all_gather_into_tensor(output_tensor=out, input_tensor=inp)

        _, _, out_info = self._tracked_info(out)
        self.assertIn(int(side.npu_stream), out_info.cross_stream_usages)

    def test_list_and_none_tolerance(self):
        outs = [torch.empty(4, device="npu") for _ in range(2)]
        inp = torch.randn(8, device="npu")
        side, _ = self._side_stream_context()

        with torch_npu.npu.stream(side):
            dist.all_gather(tensor_list=outs, tensor=inp)
            dist.all_to_all([None], [inp])

        side_id = int(side.npu_stream)
        for t in outs:
            _, _, info = self._tracked_info(t)
            self.assertIn(side_id, info.cross_stream_usages)

    def test_p2p_records_tensor(self):
        t = torch.randn(4, device="npu")
        side, _ = self._side_stream_context()

        with torch_npu.npu.stream(side):
            dist.send(t, 1)
            dist.recv(t, 0)

        _, _, info = self._tracked_info(t)
        self.assertIn(int(side.npu_stream), info.cross_stream_usages)

    def test_2_14_public_entries_covered(self):
        """The torch 2.14 public trampolines (all_gather_single,
        reduce_scatter_single) and the legacy gather/scatter/reduce
        entries are wrapped when present and record their tensors."""
        missing = [
            name for name in (
                "all_gather_single", "reduce_scatter_single",
                "gather", "scatter", "reduce",
            )
            if not hasattr(dist, name)
        ]
        if missing:
            self.skipTest(f"not present on this torch: {', '.join(missing)}")
        side, _ = self._side_stream_context()
        side_id = int(side.npu_stream)

        t = torch.randn(4, device="npu")
        with torch_npu.npu.stream(side):
            dist.reduce(t, 0)
        _, _, info = self._tracked_info(t)
        self.assertIn(side_id, info.cross_stream_usages)
        self.assertEqual(
            info.cross_stream_usages[side_id].operator, "dist [reduce]")

        inp = torch.randn(4, device="npu")
        out = torch.empty(4, device="npu")
        with torch_npu.npu.stream(side):
            dist.all_gather_single(out, inp)
        _, _, info = self._tracked_info(out)
        self.assertIn(side_id, info.cross_stream_usages)
        self.assertEqual(
            info.cross_stream_usages[side_id].operator,
            "dist [all_gather_single]")

        inp = torch.randn(4, device="npu")
        out = torch.empty(4, device="npu")
        with torch_npu.npu.stream(side):
            dist.reduce_scatter_single(out, inp)
        _, _, info = self._tracked_info(out)
        self.assertIn(side_id, info.cross_stream_usages)
        self.assertEqual(
            info.cross_stream_usages[side_id].operator,
            "dist [reduce_scatter_single]")

        g1 = torch.empty(4, device="npu")
        g2 = torch.empty(4, device="npu")
        with torch_npu.npu.stream(side):
            dist.gather(torch.randn(4, device="npu"), [g1, g2], 0)
        _, _, info = self._tracked_info(g2)
        self.assertIn(side_id, info.cross_stream_usages)
        self.assertEqual(
            info.cross_stream_usages[side_id].operator, "dist [gather]")

        s1 = torch.randn(4, device="npu")
        s2 = torch.randn(4, device="npu")
        out = torch.empty(4, device="npu")
        with torch_npu.npu.stream(side):
            dist.scatter(out, [s1, s2], 0)
        _, _, info = self._tracked_info(out)
        self.assertIn(side_id, info.cross_stream_usages)
        self.assertEqual(
            info.cross_stream_usages[side_id].operator, "dist [scatter]")

    def test_batch_isend_irecv_records_p2p_tensors(self):
        """``batch_isend_irecv`` records every ``P2POp.tensor``;
        ``isend``/``irecv`` stay unwrapped because ``P2POp`` validates
        its ``op`` by identity."""
        store_path = os.path.join(
            tempfile.gettempdir(), f"san_p2p_{os.getpid()}"
        )
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo", world_size=1, rank=0,
                init_method=f"file://{store_path}",
            )
        try:
            send_t = torch.randn(4, device="npu")
            recv_t = torch.empty(4, device="npu")
            send_op = dist.P2POp(dist.isend, send_t, 0)
            recv_op = dist.P2POp(dist.irecv, recv_t, 0)
            side, _ = self._side_stream_context()
            side_id = int(side.npu_stream)

            with torch_npu.npu.stream(side):
                dist.batch_isend_irecv([send_op, recv_op])

            _, _, send_info = self._tracked_info(send_t)
            _, _, recv_info = self._tracked_info(recv_t)
            self.assertIn(side_id, send_info.cross_stream_usages)
            self.assertIn(side_id, recv_info.cross_stream_usages)
            self.assertEqual(
                send_info.cross_stream_usages[side_id].operator,
                "dist [batch_isend_irecv]",
            )
        finally:
            dist.destroy_process_group()
            try:
                os.unlink(store_path)
            except OSError:
                pass

    def test_runtime_compensated_usage_not_recorded(self):
        """An operand the runtime recorded on another stream during
        the call is already protected and skipped; uncompensated ones
        still record."""
        inp = torch.randn(64, device="npu")
        out = torch.empty(64, device="npu")
        handler = get_event_handler()
        fake_stream_id = 12345

        def compensating_fake(output_tensor, input_tensor):
            handler._handle_record_stream(
                input_tensor.untyped_storage().data_ptr(), fake_stream_id
            )

        dist_check = get_dist_observation()
        dist.all_gather_into_tensor = compensating_fake
        dist_check.install(get_event_handler())

        side, _ = self._side_stream_context()
        side_id = int(side.npu_stream)
        with torch_npu.npu.stream(side):
            dist.all_gather_into_tensor(out, inp)

        _, _, inp_info = self._tracked_info(inp)
        _, _, out_info = self._tracked_info(out)
        self.assertEqual(len(inp_info.cross_stream_usages), 0)
        self.assertIn(side_id, out_info.cross_stream_usages)


class TestDistProtocolSelfCheck(SanitizerDistTestBase):
    def test_protocol_drift_warns_once_per_function(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dist.all_reduce(object(), object())
        self.assertTrue(
            any("protocol may have changed" in str(w.message) for w in caught)
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dist.all_reduce(object(), object())
        self.assertFalse(
            any("protocol may have changed" in str(w.message) for w in caught)
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dist.broadcast(object(), object())
        self.assertTrue(
            any("protocol may have changed" in str(w.message) for w in caught)
        )

    def test_plausible_calls_do_not_warn(self):
        t = torch.randn(4, device="npu")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dist.all_reduce(t)
            dist.broadcast(t, 0)
            dist.send(t, 1)
            dist.all_gather_into_tensor(torch.empty(4, device="npu"), t)
            dist._all_gather_base(
                output_tensor=torch.empty(4, device="npu"),
                input_tensor=t,
            )
        self.assertFalse(
            any("protocol may have changed" in str(w.message) for w in caught)
        )

    def test_spec_length_mismatch_rejected(self):
        """A spec whose position and keyword tuples differ in length is
        rejected at wrapper creation: the data-driven table stays safe
        to extend."""
        dist_check = get_dist_observation()
        with self.assertRaises(ValueError):
            dist_check._make_wrapper(
                "all_reduce", _fake_comm,
                ((0, 1), (0,), ("tensor",), ("tensor",)),
            )


class TestDistFreeTimeIntegration(SanitizerDistTestBase):
    def _cross_stream_setup(self):
        """Output allocated on a side stream, collective issued on the
        default stream: the allocation stream never waits the usage
        stream, which is exactly the free-time hazard to report."""
        inp = torch.randn(100, device="npu")
        side = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()
        side.wait_stream(default_stream)
        with torch_npu.npu.stream(side):
            out = torch.empty(100, device="npu")
        return inp, out, side, default_stream

    def test_cross_stream_comm_reported_on_flush(self):
        inp, out, side, default_stream = self._cross_stream_setup()
        out_ptr = out.untyped_storage().data_ptr()

        dist.all_gather_into_tensor(out, inp)

        reports = get_event_handler().flush_record_stream_warnings()
        out_reports = [r for r in reports if r.data_ptr == out_ptr]
        self.assertEqual(len(out_reports), 1)
        self.assertEqual(
            out_reports[0].usage.operator, "dist [all_gather_into_tensor]"
        )
        self.assertEqual(
            out_reports[0].usage_stream, int(default_stream.npu_stream)
        )

    def test_record_stream_suppresses_report(self):
        inp, out, side, default_stream = self._cross_stream_setup()
        out_ptr = out.untyped_storage().data_ptr()

        out.record_stream(default_stream)
        dist.all_gather_into_tensor(out, inp)

        reports = get_event_handler().flush_record_stream_warnings()
        out_reports = [r for r in reports if r.data_ptr == out_ptr]
        self.assertEqual(len(out_reports), 0)

    def test_wait_stream_edge_suppresses_report(self):
        inp, out, side, default_stream = self._cross_stream_setup()
        out_ptr = out.untyped_storage().data_ptr()

        dist.all_gather_into_tensor(out, inp)
        side.wait_stream(default_stream)

        reports = get_event_handler().flush_record_stream_warnings()
        out_reports = [r for r in reports if r.data_ptr == out_ptr]
        self.assertEqual(len(out_reports), 0)

    def test_cross_stream_comm_reported_on_dealloc(self):
        inp, out, side, default_stream = self._cross_stream_setup()
        out_ptr = out.untyped_storage().data_ptr()

        dist.all_gather_into_tensor(out, inp)
        del out, inp
        gc.collect()

        errors = get_event_handler().record_stream_errors
        out_errors = [e for e in errors if e.data_ptr == out_ptr]
        self.assertGreater(len(out_errors), 0)


class TestDistThirdPartyNesting(SanitizerDistTestBase):
    def test_save_replace_restore_chain(self):
        """A MindSpeed-style patch replaces the module function with a
        stub while active (cache replay, no real collective): those
        calls record nothing; restoring our wrapper resumes recording."""
        inp = torch.randn(100, device="npu")
        side = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()
        side.wait_stream(default_stream)
        with torch_npu.npu.stream(side):
            out = torch.empty(100, device="npu")
        our_wrapper = dist._all_gather_base

        def stub(*args, **kwargs):
            return None

        dist._all_gather_base = stub
        dist._all_gather_base(out, inp)
        _, _, info = self._tracked_info(out)
        self.assertEqual(len(info.cross_stream_usages), 0)

        dist._all_gather_base = our_wrapper
        dist._all_gather_base(out, inp)
        _, _, info = self._tracked_info(out)
        self.assertIn(int(default_stream.npu_stream), info.cross_stream_usages)

    def test_uninstall_leaves_active_third_party_patch(self):
        """Uninstall restores only slots still holding our wrapper: an
        active third-party patch stays in place, and the wrapper it
        restores later is a safe pass-through."""
        dist_check = get_dist_observation()
        our_wrapper = dist.all_reduce

        def third_party(*args, **kwargs):
            return None

        dist.all_reduce = third_party
        dist_check.uninstall()
        self.assertIs(dist.all_reduce, third_party)

        dist.all_reduce = our_wrapper
        t = torch.randn(4, device="npu")
        self.assertIsNone(dist.all_reduce(t))
        _, _, info = self._tracked_info(t)
        self.assertEqual(len(info.cross_stream_usages), 0)

    def test_reinstall_after_patch_restore_reuses_wrapper(self):
        """After an uninstall under an active patch, the wrapper the
        patch restores is reused on reinstall, not rewrapped."""
        dist_check = get_dist_observation()
        wrapper = dist.all_reduce

        def third_party(*args, **kwargs):
            return None

        dist.all_reduce = third_party
        dist_check.uninstall()
        dist.all_reduce = wrapper

        dist_check.install(get_event_handler())
        self.assertIs(dist.all_reduce, wrapper)
        self.assertIs(dist_check.wrappers["all_reduce"], wrapper)
        self.assertIs(wrapper.handler, get_event_handler())

        t = torch.randn(4, device="npu")
        side, _ = self._side_stream_context()
        with torch_npu.npu.stream(side):
            dist.all_reduce(t)
        _, _, info = self._tracked_info(t)
        self.assertIn(int(side.npu_stream), info.cross_stream_usages)

    def test_reinstall_over_active_stub_records_once(self):
        """Re-installing over an active third-party stub: the stale
        wrapper stays inert, so one collective advances the
        pseudo-launch clock exactly once."""
        dist_check = get_dist_observation()
        wrapper = dist.all_reduce

        def stub(*args, **kwargs):
            return wrapper(*args, **kwargs)

        dist.all_reduce = stub
        dist_check.uninstall()
        self.assertIsNone(wrapper.handler)
        dist_check.install(get_event_handler())

        t = torch.randn(4, device="npu")
        side, _ = self._side_stream_context()
        handler = get_event_handler()
        seq_before = handler.seq_num
        with torch_npu.npu.stream(side):
            dist.all_reduce(t)
        self.assertEqual(handler.seq_num - seq_before, 1)
        _, _, info = self._tracked_info(t)
        self.assertIn(int(side.npu_stream), info.cross_stream_usages)


class TestSanitizerExitFlush(TestCase):
    def setUp(self):
        reset_sanitizer()
        get_sanitizer_module()._atexit_flush_registered = False

    def tearDown(self):
        reset_sanitizer()
        get_sanitizer_module()._atexit_flush_registered = False
        get_engine_observation().uninstall()
        get_dist_observation().uninstall()

    def test_enable_registers_exit_flush_once(self):
        """enable → reset → enable must not add a second exit callback."""
        sanitizer = get_sanitizer_module()
        with mock.patch("atexit.register") as fake_register:
            setup_sanitizer()
            self.assertEqual(fake_register.call_count, 1)
            self.assertIs(
                fake_register.call_args[0][0],
                sanitizer._flush_record_stream_warnings_atexit,
            )

            reset_sanitizer()
            setup_sanitizer()
            self.assertEqual(fake_register.call_count, 1)

    def test_exit_flush_targets_live_handler(self):
        """The exit hook reads the handler at exit time: no handler
        means no-op, and whichever handler is live gets flushed."""
        sanitizer = get_sanitizer_module()
        flush = sanitizer._flush_record_stream_warnings_atexit

        flush()

        with mock.patch.object(
            sanitizer.npu_sanitizer, "event_handler"
        ) as fake_handler:
            flush()
        fake_handler.flush_record_stream_warnings.assert_called_once_with()

    def test_enable_is_idempotent(self):
        """A second enable() must not stack handlers or dispatch modes:
        enable() has its own re-entrancy guard."""
        sanitizer = get_sanitizer_module()
        setup_sanitizer()
        handler_before = sanitizer.npu_sanitizer.event_handler
        dispatch_before = sanitizer.npu_sanitizer.dispatch
        sanitizer.npu_sanitizer.enable()
        self.assertIs(sanitizer.npu_sanitizer.event_handler, handler_before)
        self.assertIs(sanitizer.npu_sanitizer.dispatch, dispatch_before)


if __name__ == "__main__":
    run_tests()
