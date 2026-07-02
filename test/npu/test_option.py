import os
import sys
import subprocess
import unittest
from functools import wraps

import torch
import torch_npu

import torch_npu.npu.utils as utils

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices, SkipIfNotGteCANNVersion


class TestOption(TestCase):

    def test_option_pm(self):
        option = {"ACL_PRECISION_MODE": "allow_fp32_to_fp16"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_osim(self):
        option = {"ACL_OP_SELECT_IMPL_MODE": "high_precision"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_ofi(self):
        option = {"ACL_OPTYPELIST_FOR_IMPLMODE": "Conv2d"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_odl(self):
        option = {"ACL_OP_DEBUG_LEVEL": "2"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_occm(self):
        option = {"ACL_OP_COMPILER_CACHE_MODE": "enable"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_dd(self):
        option = {"ACL_DEBUG_DIR": "test1"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_occd(self):
        option = {"ACL_OP_COMPILER_CACHE_DIR": "test"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_an(self):
        option = {"ACL_AICORE_NUM": "1"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_pme(self):
        option = {"ACL_PRECISION_MODE": "500"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_osime(self):
        option = {"ACL_OP_SELECT_IMPL_MODE": "100"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_dle(self):
        option = {"ACL_OP_DEBUG_LEVEL": "300"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_occme(self):
        option = {"ACL_OP_COMPILER_CACHE_MODE": "2"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_ane(self):
        option = {"ACL_AICORE_NUM": "at"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_fa(self):
        option = {"FORCE_ACLNN_OP_LIST": "index"}
        self.assertIsNone(torch.npu.set_option(option))


class TestAclOpInitMode(TestCase):

    ACLNN_WARN = "ACL_OP_INIT_MODE=0 or 1 is not supported on this device."
    INVALID_VALUE_WARN = "Get env ACL_OP_INIT_MODE not in [0, 1, 2]"

    def _run_subprocess(self, env_val):
        env_line = f"os.environ['ACL_OP_INIT_MODE']='{env_val}'" if env_val is not None else ""
        test_script = f"import os; {env_line}; import torch; import torch_npu; torch_npu.npu.set_device(0)"
        result = subprocess.run(
            [sys.executable, '-c', test_script],
            capture_output=True, text=True
        )
        return result.stderr

    @SupportedDevices(['Ascend950'])
    def test_mode0_on_ascend950(self):
        stderr = self._run_subprocess("0")
        self.assertIn(self.ACLNN_WARN, stderr,
                      "ACL_OP_INIT_MODE=0 should be auto-corrected to 2 on Ascend950")

    @SupportedDevices(['Ascend950'])
    def test_mode1_on_ascend950(self):
        stderr = self._run_subprocess("1")
        self.assertIn(self.ACLNN_WARN, stderr,
                      "ACL_OP_INIT_MODE=1 should be auto-corrected to 2 on Ascend950")

    @SupportedDevices(['Ascend910B'])
    def test_mode0_on_non_ascend950(self):
        stderr = self._run_subprocess("0")
        self.assertNotIn(self.ACLNN_WARN, stderr,
                         "ACL_OP_INIT_MODE=0 should stay 0 on non-Ascend950 device")

    def test_mode_invalid(self):
        stderr = self._run_subprocess("999")
        self.assertIn(self.INVALID_VALUE_WARN, stderr)


class TestAclInitConfigPath(TestCase):

    LACKS_DEFAULT_DEVICE_MSG = "lacks 'defaultDevice'"
    HAS_DEFAULT_DEVICE_MSG = "contains 'defaultDevice'"
    PARSE_FAILED_MSG = "Failed to parse user acl json"
    INVALID_PATH_MSG = "is invalid"
    OPEN_FAILED_MSG = "Failed to open user acl json"

    def _make_temp_json(self, content):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        tmp.write(content)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def _run_with_env(self, json_path):
        test_script = (
            f"import os; "
            f"os.environ['TORCH_ACL_INIT_CONFIG_PATH']='{json_path}'; "
            f"import torch; import torch_npu; torch_npu.npu.set_device(0)"
        )
        result = subprocess.run(
            [sys.executable, '-c', test_script],
            capture_output=True, text=True
        )
        return result

    # ---- 设备无关：路径无效/解析失败 ----

    def test_invalid_json(self):
        """非法 JSON 格式应抛 RuntimeError"""
        json_path = self._make_temp_json('not a json{{{')
        ret = self._run_with_env(json_path)
        self.assertNotEqual(ret.returncode, 0)
        self.assertIn(self.PARSE_FAILED_MSG, ret.stderr,
                      "Invalid JSON should raise RuntimeError")

    def test_nonexistent_path(self):
        """不存在的路径应抛 RuntimeError"""
        json_path = '/tmp/nonexistent_acl_config_for_test.json'
        ret = self._run_with_env(json_path)
        self.assertNotEqual(ret.returncode, 0)
        self.assertIn(self.INVALID_PATH_MSG, ret.stderr,
                      "Non-existent path should raise RuntimeError")

    def test_empty_json_file(self):
        """空 JSON 文件应抛 RuntimeError"""
        json_path = self._make_temp_json('')
        ret = self._run_with_env(json_path)
        self.assertNotEqual(ret.returncode, 0)
        self.assertIn(self.PARSE_FAILED_MSG, ret.stderr,
                      "Empty JSON file should raise RuntimeError")

    # ---- non-lazy 模式（CANN < 8.3.RC1） ----

    @staticmethod
    def _skipIfLazy(fn):
        @wraps(fn)
        def wrapper(slf, *args, **kwargs):
            if utils._is_gte_cann_version('8.3.RC1'):
                raise unittest.SkipTest(
                    "Test only for non-lazy set_device mode (CANN < 8.3.RC1)")
            return fn(slf, *args, **kwargs)
        return wrapper

    @_skipIfLazy
    def test_valid_json_non_lazy(self):
        """non-lazy 模式：不含 defaultDevice 的用户 JSON 正常使用"""
        json_path = self._make_temp_json('{"dump":{"dump_scene":"lite_exception"}}')
        ret = self._run_with_env(json_path)
        self.assertEqual(ret.returncode, 0,
                         f"Valid JSON in non-lazy should succeed, got: {ret.stderr}")

    @_skipIfLazy
    def test_valid_json_with_default_device_in_non_lazy(self):
        """non-lazy 模式：含 defaultDevice 应抛 RuntimeError"""
        json_path = self._make_temp_json(
            '{"dump":{"dump_scene":"lite_exception"},"defaultDevice":{"default_device":"0"}}'
        )
        ret = self._run_with_env(json_path)
        self.assertNotEqual(ret.returncode, 0)
        self.assertIn(self.HAS_DEFAULT_DEVICE_MSG, ret.stderr,
                      "Non-lazy mode with defaultDevice should raise RuntimeError")

    # ---- lazy 模式（CANN >= 8.3.RC1） ----

    @SkipIfNotGteCANNVersion('8.3.RC1')
    def test_valid_json_with_default_device_in_lazy(self):
        """lazy 模式：含合法 defaultDevice 正常使用"""
        json_path = self._make_temp_json(
            '{"dump":{"dump_scene":"lite_exception"},"defaultDevice":{"default_device":"0"}}'
        )
        ret = self._run_with_env(json_path)
        self.assertEqual(ret.returncode, 0,
                         f"Valid JSON in lazy should succeed, got: {ret.stderr}")

    @SkipIfNotGteCANNVersion('8.3.RC1')
    def test_json_without_default_device_in_lazy(self):
        """lazy 模式：不含 defaultDevice 应抛 RuntimeError"""
        json_path = self._make_temp_json('{"dump":{"dump_scene":"lite_exception"}}')
        ret = self._run_with_env(json_path)
        self.assertNotEqual(ret.returncode, 0)
        self.assertIn(self.LACKS_DEFAULT_DEVICE_MSG, ret.stderr,
                      "Lazy mode without defaultDevice should raise RuntimeError")

    @SkipIfNotGteCANNVersion('8.3.RC1')
    def test_json_wrong_default_device_value_in_lazy(self):
        """lazy 模式：default_device 值不为 '0' 应抛 RuntimeError"""
        json_path = self._make_temp_json(
            '{"defaultDevice":{"default_device":"1"}}'
        )
        ret = self._run_with_env(json_path)
        self.assertNotEqual(ret.returncode, 0)
        self.assertIn('requires', ret.stderr,
                      "default_device='1' should raise RuntimeError")

    @SkipIfNotGteCANNVersion('8.3.RC1')
    def test_json_default_device_wrong_type(self):
        """lazy 模式：defaultDevice 非 object 应抛 RuntimeError"""
        json_path = self._make_temp_json(
            '{"defaultDevice":"not_an_object","dump":{"dump_scene":"lite_exception"}}'
        )
        ret = self._run_with_env(json_path)
        self.assertNotEqual(ret.returncode, 0)
        self.assertIn(self.LACKS_DEFAULT_DEVICE_MSG, ret.stderr,
                      "defaultDevice as string should raise RuntimeError")


if __name__ == "__main__":
    run_tests()
