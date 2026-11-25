# Owner(s): ["oncall: profiler"]
import os
import shutil
from unittest import mock

from torch_npu._C._profiler import ProfilerActivity
from torch_npu.npu import Event
from torch_npu.profiler import supported_activities
from torch_npu.profiler._profiler_path_creator import ProfPathCreator
from torch_npu.profiler.profiler_interface import (
    _disable_event_record,
    _enable_event_record,
    _ProfInterface,
)
from torch_npu.testing.testcase import run_tests, TestCase

import torch


class TestActionController(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.prof_dir = "./result_dir"
        cls.namespace = "torch_npu.profiler.profiler_interface"

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.prof_dir):
            shutil.rmtree(cls.prof_dir)

    def setUp(self):
        self.prof_if = _ProfInterface()
        ProfPathCreator().init(dir_name=self.prof_dir)

    def test_init_trace(self):
        self.prof_if.custom_trace_id_callback = lambda: "trace_0"
        with mock.patch(self.namespace + "._init_profiler") as mock_func:
            self.prof_if.init_trace()
            self.assertEqual(1, mock_func.call_count)
            self.assertTrue(os.path.exists(self.prof_dir))
            self.assertEqual("trace_0", self.prof_if.trace_id)

    def test_start_trace(self):
        with (
            mock.patch(
                self.namespace + ".NpuProfilerConfig", return_value="config_obj"
            ),
            mock.patch(self.namespace + "._get_syscnt_enable", return_value=True),
            mock.patch(self.namespace + "._get_freq", return_value=100),
            mock.patch(self.namespace + "._get_syscnt", return_value=10000),
            mock.patch(self.namespace + "._get_monotonic", return_value=20000),
            mock.patch(self.namespace + "._start_profiler") as mock_start,
        ):
            self.prof_if.start_trace()
            self.assertEqual(True, self.prof_if.syscnt_enable)
            self.assertEqual(100, self.prof_if.freq)
            self.assertEqual(10000, self.prof_if.start_cnt)
            self.assertEqual(20000, self.prof_if.start_monotonic)
            mock_start.assert_called_once_with("config_obj", supported_activities())

    def test_start_trace_flushes_preset_metadata(self):
        preset_metadata = {"preset_key": '{"value": 1}'}
        metadata_json_adder = mock.Mock()
        self.prof_if.preset_metadata = preset_metadata
        self.prof_if.metadata_json_adder = metadata_json_adder
        with (
            mock.patch(
                self.namespace + ".NpuProfilerConfig", return_value="config_obj"
            ),
            mock.patch(self.namespace + "._start_profiler"),
            mock.patch(self.namespace + "._get_syscnt_enable", return_value=False),
            mock.patch(self.namespace + "._get_monotonic", return_value=100),
        ):
            self.prof_if.start_trace()

        metadata_json_adder.assert_called_once_with("preset_key", '{"value": 1}')

    def test_stop_trace(self):
        with mock.patch(self.namespace + "._stop_profiler") as mock_stop:
            self.prof_if.stop_trace()
            mock_stop.assert_called_once()

        from itertools import product
        from torch_npu.profiler._profiler_action_controller import ProfActionController
        from torch_npu.profiler.profiler import profile as LegacyProfile
        from torch_npu.profiler.scheduler import ProfilerAction

        stop_errors = (RuntimeError("stop failure"), KeyboardInterrupt(), SystemExit())
        for stop_error, cleanup_error, public in product(stop_errors, (None, OSError("finalize failure")), (False, True)):
            order = mock.Mock()
            order.stop.side_effect, order.finalize.side_effect = stop_error, cleanup_error
            with (
                self.subTest(stop_error=stop_error, cleanup_error=cleanup_error, public=public),
                mock.patch.multiple(self.namespace, _stop_profiler=order.stop, _finalize_profiler=order.finalize,
                                    print_error_msg=mock.DEFAULT, ProfPathCreator=mock.DEFAULT) as patched,
                mock.patch(self.namespace + ".ProfilerLogger.destroy"),
                mock.patch.multiple(self.prof_if, _dump_profiler_info=mock.DEFAULT, _dump_metadata=mock.DEFAULT) as dump,
                mock.patch("torch_npu.profiler.analysis.prof_common_func._utils.print_error_msg") as public_error,
            ):
                self.prof_if.activities = set()
                ready = mock.Mock()
                controller = ProfActionController(mock.Mock(), self.prof_if, ready)
                legacy = mock.Mock(record_steps=False, action_controller=controller,
                                   current_action=ProfilerAction.RECORD_AND_SAVE, execution_trace_observer=None)

                def action():
                    if public:
                        return LegacyProfile.stop(legacy)
                    return controller.transit_action(legacy.current_action, None)

                if public and isinstance(stop_error, Exception):
                    self.assertIsNone(action())
                    public_error.assert_called_once_with("Call stop failed. Exception: stop failure")
                else:
                    with self.assertRaises(type(stop_error)) as raised:
                        action()
                    self.assertIs(raised.exception, stop_error)
                    public_error.assert_not_called()
                self.assertEqual([c[0] for c in order.mock_calls], ["stop", "finalize"])
                ready.assert_not_called()
                if cleanup_error:
                    patched["print_error_msg"].assert_called_once_with("Profiler finalization failed: finalize failure")
                else:
                    dump["_dump_metadata"].assert_called_once()

    def test_start_and_stop_trace_with_flops_and_msprof_tx(self):
        self.prof_if.with_flops = True
        self.prof_if.experimental_config._msprof_tx = True
        with (
            mock.patch(self.namespace + "._start_profiler"),
            mock.patch(self.namespace + "._stop_profiler"),
            mock.patch(self.namespace + ".FlopsHookManager.install") as mock_install,
            mock.patch(self.namespace + ".FlopsHookManager.uninstall") as mock_uninstall,
        ):
            self.prof_if.start_trace()
            self.prof_if.stop_trace()
        mock_install.assert_called_once()
        mock_uninstall.assert_called_once()

    def test_start_and_stop_trace_with_flops_without_msprof_tx_skips_flops_hook(self):
        self.prof_if.with_flops = True
        with (
            mock.patch(self.namespace + "._start_profiler"),
            mock.patch(self.namespace + "._stop_profiler"),
            mock.patch(self.namespace + ".FlopsHookManager.install") as mock_install,
            mock.patch(self.namespace + ".FlopsHookManager.uninstall") as mock_uninstall,
        ):
            self.prof_if.start_trace()
            self.prof_if.stop_trace()
        mock_install.assert_not_called()
        mock_uninstall.assert_not_called()

    def test_finalize_trace(self):
        with (
            mock.patch(self.namespace + "._init_profiler"),
            mock.patch(self.namespace + "._finalize_profiler") as mock_finalize,
        ):
            self.prof_if.metadata = {"key": "val"}
            self.prof_if.init_trace()
            self.prof_if.finalize_trace()
            mock_finalize.assert_called_once()
            self.assertTrue(self._check_profiler_info_json(self.prof_if.prof_path))
            self.assertTrue(self._check_metadata_json(self.prof_if.prof_path))

    def test_analyse(self):
        with mock.patch(
            "torch_npu.profiler.analysis._npu_profiler.NpuProfiler.analyse"
        ) as mock_analyse:
            self.prof_if.analyse()
            mock_analyse.assert_called_once()

    def test_supported_activities(self):
        activities = set(supported_activities())
        self.assertEqual(2, len(activities))
        self.assertTrue(ProfilerActivity.CPU in activities)
        self.assertTrue(ProfilerActivity.NPU in activities)

    def test_create_trace_id_should_return_default_when_callback_is_none(self):
        self.prof_if.custom_trace_id_callback = None
        trace_id = self.prof_if.create_trace_id()
        self.assertIsInstance(trace_id, str)
        self.assertEqual(32, len(trace_id))

    def test_create_trace_id_should_return_callback_result_when_callback_is_valid(self):
        self.prof_if.custom_trace_id_callback = lambda: "custom_trace_id"
        trace_id = self.prof_if.create_trace_id()
        self.assertEqual("custom_trace_id", trace_id)

    def test_create_trace_id_should_return_default_when_callback_is_not_callable(self):
        self.prof_if.custom_trace_id_callback = "not_callable"
        trace_id = self.prof_if.create_trace_id()
        self.assertIsInstance(trace_id, str)
        self.assertEqual(32, len(trace_id))

    def test_create_trace_id_should_return_default_when_callback_returns_non_string(
        self,
    ):
        self.prof_if.custom_trace_id_callback = lambda: 12345
        trace_id = self.prof_if.create_trace_id()
        self.assertIsInstance(trace_id, str)
        self.assertEqual(32, len(trace_id))

    def test_create_trace_id_should_return_default_when_callback_raises_exception(self):
        def bad_callback():
            raise RuntimeError("callback error")

        self.prof_if.custom_trace_id_callback = bad_callback
        trace_id = self.prof_if.create_trace_id()
        self.assertIsInstance(trace_id, str)
        self.assertEqual(32, len(trace_id))

    def test_create_trace_id_should_return_default_when_callback_returns_too_long_string(
        self,
    ):
        long_str = "a" * (self.prof_if.MAX_TRACE_ID_LEN + 1)
        self.prof_if.custom_trace_id_callback = lambda: long_str
        trace_id = self.prof_if.create_trace_id()
        self.assertIsInstance(trace_id, str)
        self.assertEqual(32, len(trace_id))

    def test_event_record_should_have_return_true_attr_when_enable_record(self):
        _enable_event_record()
        self.assertTrue(hasattr(Event.record, "origin_func"))
        self.assertTrue(hasattr(Event.wait, "origin_func"))
        self.assertTrue(hasattr(Event.query, "origin_func"))
        self.assertTrue(hasattr(Event.elapsed_time, "origin_func"))
        self.assertTrue(hasattr(Event.synchronize, "origin_func"))
        _disable_event_record()
        self.assertFalse(hasattr(Event.record, "origin_func"))
        self.assertFalse(hasattr(Event.wait, "origin_func"))
        self.assertFalse(hasattr(Event.query, "origin_func"))
        self.assertFalse(hasattr(Event.elapsed_time, "origin_func"))
        self.assertFalse(hasattr(Event.synchronize, "origin_func"))

    def _check_profiler_info_json(self, prof_path: str) -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_id = torch.distributed.get_rank()
            path = os.path.join(
                os.path.realpath(prof_path), f"profiler_info_{rank_id}.json"
            )
        else:
            path = os.path.join(os.path.realpath(prof_path), "profiler_info.json")
        return os.path.exists(path)

    def _check_metadata_json(self, prof_path: str) -> bool:
        path = os.path.join(os.path.realpath(prof_path), "profiler_metadata.json")
        return os.path.exists(path)


if __name__ == "__main__":
    run_tests()
