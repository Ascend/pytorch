import unittest
import os

from torch_npu.profiler.analysis.prof_common_func._utils import check_msprof_env
from torch_npu.profiler.analysis.prof_common_func._constant import Constant


class TestCheckMsprofEnv(unittest.TestCase):

    def test_check_msprof_env_should_return_false_when_only_static_env_set(self):
        if hasattr(check_msprof_env, '_called'):
            delattr(check_msprof_env, '_called')
        if hasattr(check_msprof_env, '_cached_result'):
            delattr(check_msprof_env, '_cached_result')
        os.environ[Constant.MSPROF_STATIC_ENV] = "test_static"
        result = check_msprof_env()
        self.assertFalse(result)
        del os.environ[Constant.MSPROF_STATIC_ENV]

    def test_check_msprof_env_should_return_false_when_only_dynamic_env_set(self):
        if hasattr(check_msprof_env, '_called'):
            delattr(check_msprof_env, '_called')
        if hasattr(check_msprof_env, '_cached_result'):
            delattr(check_msprof_env, '_cached_result')
        os.environ[Constant.MSPROF_DYNAMIC_ENV] = "test_dynamic"
        result = check_msprof_env()
        self.assertFalse(result)
        del os.environ[Constant.MSPROF_DYNAMIC_ENV]

    def test_check_msprof_env_should_return_false_when_both_envs_set(self):
        if hasattr(check_msprof_env, '_called'):
            delattr(check_msprof_env, '_called')
        if hasattr(check_msprof_env, '_cached_result'):
            delattr(check_msprof_env, '_cached_result')
        os.environ[Constant.MSPROF_STATIC_ENV] = "test_static"
        os.environ[Constant.MSPROF_DYNAMIC_ENV] = "test_dynamic"
        result = check_msprof_env()
        self.assertFalse(result)
        del os.environ[Constant.MSPROF_STATIC_ENV]
        del os.environ[Constant.MSPROF_DYNAMIC_ENV]

    def test_check_msprof_env_should_return_true_when_no_env_set(self):
        if hasattr(check_msprof_env, '_called'):
            delattr(check_msprof_env, '_called')
        if hasattr(check_msprof_env, '_cached_result'):
            delattr(check_msprof_env, '_cached_result')
        if Constant.MSPROF_STATIC_ENV in os.environ:
            del os.environ[Constant.MSPROF_STATIC_ENV]
        if Constant.MSPROF_DYNAMIC_ENV in os.environ:
            del os.environ[Constant.MSPROF_DYNAMIC_ENV]
        result = check_msprof_env()
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()