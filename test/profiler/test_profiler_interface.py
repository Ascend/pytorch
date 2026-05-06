# Owner(s): ["oncall: profiler"]
import os
import shutil
from unittest import mock

from torch_npu._C._profiler import ProfilerActivity
from torch_npu.npu import Event
from torch_npu.profiler import supported_activities
from torch_npu.profiler._profiler_path_creator import ProfPathCreator
from torch_npu.profiler.analysis.prof_common_func._cann_package_manager import (
    CannPackageManager,
)
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
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

    def test_stop_trace(self):
        with mock.patch(self.namespace + "._stop_profiler") as mock_stop:
            self.prof_if.stop_trace()
            mock_stop.assert_called_once()

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

    def _check_params(self):
        CannPackageManager.SUPPORT_EXPORT_DB = False
        self.prof_if.activities = set(supported_activities())
        self.prof_if.experimental_config.export_type = Constant.Db
        with self.assertExpectedRaises(RuntimeError):
            self.prof_if._check_params()


if __name__ == "__main__":
    run_tests()
