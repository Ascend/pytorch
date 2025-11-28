import tempfile
import os
import warnings
import logging
from unittest.mock import patch

import torch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.utils._step import (
    PerfDumpState,
    _is_loss_module,
    _validate_path,
    _get_perf_dump_path,
    delete_pref_pt_logs,
    _get_uuid,
    _setup_logger,
    _perf_dump_decorator,
    _prase_asd_config
)


class TestStep(TestCase):

    def test_parse_asd_config_invalid(self):
        with self.assertRaises(ValueError):
            _prase_asd_config({"with_checksum": "invalid"})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _prase_asd_config({"cooldown": "invalid"})
            self.assertTrue(len(w) > 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _prase_asd_config({"strikes_num": "invalid"})
            self.assertTrue(len(w) > 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _prase_asd_config({"strikes_window": "invalid"})
            self.assertTrue(len(w) > 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _prase_asd_config({"checksum_cooldown": "invalid"})
            self.assertTrue(len(w) > 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _prase_asd_config({"upper_thresh1": "invalid"})
            self.assertTrue(len(w) > 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _prase_asd_config({"upper_thresh2": "invalid"})
            self.assertTrue(len(w) > 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _prase_asd_config({"grad_sample_interval": "invalid"})
            self.assertTrue(len(w) > 0)

    def test_perf_dump_decorator_not_initialized(self):
        with patch('torch.npu.is_initialized', return_value=False):
            class MockModule:
                def __call__(self, *args, **kwargs):
                    return "mock_result"

            mock_module = MockModule()
            decorated_func = _perf_dump_decorator(mock_module.__call__)
            result = decorated_func(mock_module)
            self.assertEqual(result, "mock_result")

    def test_setup_logger(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _setup_logger("test_logger", tmp_path)
            logger = logging.getLogger("test_logger")
            self.assertIsNotNone(logger)
            self.assertTrue(len(logger.handlers) > 0)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_get_uuid_missing_env(self):
        if "MASTER_ADDR" in os.environ:
            del os.environ["MASTER_ADDR"]
        if "MASTER_PORT" in os.environ:
            del os.environ["MASTER_PORT"]

        result = _get_uuid()
        self.assertEqual(result, "127.0.0.1_8888")

    def test_get_uuid_valid_env(self):
        os.environ["MASTER_ADDR"] = "192.168.1.1"
        os.environ["MASTER_PORT"] = "8080"

        result = _get_uuid()
        self.assertEqual(result, "192.168.1.1_8080")

        del os.environ["MASTER_ADDR"]
        del os.environ["MASTER_PORT"]

    def test_is_loss_module(self):
        loss_module = torch.nn.CrossEntropyLoss()
        self.assertTrue(_is_loss_module(loss_module))

        regular_module = torch.nn.Linear(10, 5)
        self.assertFalse(_is_loss_module(regular_module))

    def test_delete_pref_pt_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "perf_pt_test_0.log")
            with open(test_file, "w") as f:
                f.write("test content")

            delete_pref_pt_logs(tmpdir, "0")
            self.assertFalse(os.path.exists(test_file))

    def test_get_perf_dump_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(RuntimeError):
                _get_perf_dump_path()

            old_path = os.environ.get("PERF_DUMP_PATH")
            os.environ["PERF_DUMP_PATH"] = tmpdir

            try:
                result = _get_perf_dump_path()
                self.assertEqual(result, tmpdir)
            finally:
                if old_path is not None:
                    os.environ["PERF_DUMP_PATH"] = old_path
                else:
                    os.environ.pop("PERF_DUMP_PATH", None)

    def test_validate_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _validate_path(tmpdir)
            self.assertTrue(result)

        result = _validate_path("/non/existent/path")
        self.assertFalse(result)

    def test_perf_dump_state_functionality(self):
        state = PerfDumpState()
        self.assertEqual(state.module_dict, {})
        self.assertTrue(state.is_outer_call)
        self.assertEqual(state.log_file_name, "")
        self.assertIsNone(state.last_time)
        self.assertFalse(state.has_log)
        self.assertEqual(state.local_uuid, "")
        self.assertEqual(state.uuid, "")

        class MockModule:
            def named_modules(self):
                return[("a", self), ("b", MockModule())]

        mock_module = MockModule()
        state.add_module_dict(mock_module)
        self.assertIn(mock_module, state.module_dict)
        self.assertIsInstance(state.module_dict[mock_module], list)

        child_module = MockModule()
        state.module_dict[mock_module] = [child_module]
        self.assertTrue(state.is_child_module(child_module))
        self.assertFalse(state.is_child_module(MockModule()))


if __name__ == "__main__":
    run_tests()