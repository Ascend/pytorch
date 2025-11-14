import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

from torch_npu.utils._path_manager import PathManager
import torch_npu.profiler.analysis.prof_common_func._log as _log
from torch_npu.testing.testcase import TestCase, run_tests


class TestLog(TestCase):
    def test_set_level(self):
        output_dir = '/tmp/test_logs'
        _log.ProfilerLogger.init(output_dir)

        new_level = logging.DEBUG
        _log.ProfilerLogger.set_level(new_level)

        logger = _log.ProfilerLogger.get_instance()
        self.assertEqual(logger.level, new_level)

        for handler in logger.handlers:
            self.assertEqual(handler.level, new_level)

    def test_init_logger(self):
        output_dir = '/tmp/test_logs'
        _log.ProfilerLogger.init(output_dir)

        logger = _log.ProfilerLogger.get_instance()
        self.assertIsNotNone(logger)

        log_dir = os.path.join(output_dir, _log.ProfilerLogger.DEFAULT_LOG_DIR)
        self.assertTrue(os.path.exists(log_dir))
        self.assertGreater(len(logger.handlers), 0)
        self.assertEqual(logger.level, _log.ProfilerLogger.DEFAULT_LOG_LEVEL)

    def test_get_instance_before_init(self):
        _log.ProfilerLogger._instance = None
        with self.assertRaises(RuntimeError):
            _log.ProfilerLogger.get_instance()


if __name__ == "__main__":
    run_tests()