# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Backend.register_backend 接口功能正确性
API 名称：torch.distributed.Backend.register_backend
API 签名：Backend.register_backend(name, func, extended_api=False, devices=None) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | devices=None 默认告警与非空设备列表 | 已覆盖（test_register_basic_and_devices） |
| 枚举选项 | extended_api=False/True；devices=None/str/list | 已覆盖（test_register_basic_and_devices） |
| 参数类型 | name 字符串、callable、bool、str/list[str] | 已覆盖（test_register_basic_and_devices、test_invalid_args） |
| 传参与不传参 | 省略与显式传入 extended_api / devices | 已覆盖（test_register_basic_and_devices） |
| 等价类/边界值 | 大小写混合名称、重复覆盖注册、已有设备不覆写 | 已覆盖（test_case_normalization_and_override） |
| 正常传参场景 | 注册后属性/列表/映射/插件表均生效 | 已覆盖（test_register_basic_and_devices） |
| 异常传参场景 | 缺必填参数或非字符串 name 抛错 | 已覆盖（test_invalid_args） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（纯 Python 后端注册表，无 Tensor 输入） |

未覆盖项及原因：
- 混合设备类型：Backend.register_backend 为纯 Python 后端注册接口，不接收 Tensor 参数。
- 多卡 HCCL 通信：本接口仅操作进程内后端注册表，无需多卡通信初始化。

注意：本接口为纯 Python 全局后端注册表操作，无浮点 Tensor 输出；
     所有用例在 tearDown 中完整恢复 Backend 全局注册表状态，确保用例间隔离。
"""
import os
import unittest
import warnings

import torch
import torch.distributed as dist
from torch.distributed import Backend, ProcessGroup

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Set RUN_SUPPLEMENT_API_TESTS=1 to run supplement API tests"
DEVICE_NAME = "cpu"
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
        DEVICE_NAME = "npu" if torch.npu.is_available() else None
    except (ImportError, AttributeError):
        DEVICE_NAME = None
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    from unittest import TestCase
    run_tests = unittest.main


def _dummy_creator(*args, **kwargs):
    return (args, kwargs)


def _dummy_creator_alt(*args, **kwargs):
    return ("alt", args, kwargs)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestBackendRegisterBackend(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self._orig_list = list(Backend.backend_list)
        self._orig_dicts = {
            "default_device_backend_map": dict(Backend.default_device_backend_map),
            "backend_capability": {k: list(v) for k, v in Backend.backend_capability.items()},
            "backend_type_map": dict(Backend.backend_type_map),
            "_plugins": dict(Backend._plugins),
        }
        self._added_attrs = []

    def tearDown(self):
        Backend.backend_list[:] = self._orig_list
        for attr, orig in self._orig_dicts.items():
            table = getattr(Backend, attr)
            table.clear()
            table.update(orig)
        for attr in self._added_attrs:
            if hasattr(Backend, attr):
                delattr(Backend, attr)
        super().tearDown()

    def _register(self, name, func=_dummy_creator, **kwargs):
        upper = name.upper()
        if not hasattr(Backend, upper):
            self._added_attrs.append(upper)
        return Backend.register_backend(name, func, **kwargs)

    def test_register_basic_and_devices(self):
        name = "ut_custom_basic"
        ret = self._register(name, devices=["ut_dev_a"])
        self.assertIsNone(ret)
        self.assertEqual(getattr(Backend, name.upper()), name.lower())
        self.assertEqual(Backend(name.upper()), name.lower())
        self.assertIn(name.lower(), Backend.backend_list)
        self.assertEqual(Backend.backend_type_map[name.lower()], ProcessGroup.BackendType.CUSTOM)
        self.assertEqual(Backend.backend_capability[name.lower()], ["ut_dev_a"])
        self.assertEqual(Backend.default_device_backend_map["ut_dev_a"], name.lower())
        plugin = Backend._plugins[name.upper()]
        self.assertIs(plugin.creator_fn, _dummy_creator)
        self.assertFalse(plugin.extended_api)

        self._register("ut_custom_ext", extended_api=True, devices="ut_single_dev")
        self.assertTrue(Backend._plugins["UT_CUSTOM_EXT"].extended_api)
        self.assertEqual(Backend.backend_capability["ut_custom_ext"], ["ut_single_dev"])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._register("ut_custom_nodev")
        self.assertTrue(any("unspecified" in str(w.message) for w in caught))
        caps = Backend.backend_capability["ut_custom_nodev"]
        self.assertIn("cpu", caps)
        self.assertIn(self.device_name, caps)

    def test_case_normalization_and_override(self):
        name = "Ut_MixedCase_Backend"
        orig_cpu_backend = Backend.default_device_backend_map.get("cpu")
        self._register(name, extended_api=False, devices=["cpu", "ut_dev_mix"])
        self.assertEqual(getattr(Backend, name.upper()), name.lower())
        self.assertEqual(Backend.backend_list.count(name.lower()), 1)
        self.assertEqual(Backend.default_device_backend_map.get("cpu"), orig_cpu_backend)
        self.assertEqual(Backend.default_device_backend_map.get("ut_dev_mix"), name.lower())

        self._register(name.lower(), _dummy_creator_alt, extended_api=True, devices=["ut_dev_mix2"])
        self.assertEqual(Backend.backend_list.count(name.lower()), 1)
        plugin = Backend._plugins[name.upper()]
        self.assertIs(plugin.creator_fn, _dummy_creator_alt)
        self.assertTrue(plugin.extended_api)
        self.assertEqual(Backend.backend_capability[name.lower()], ["ut_dev_mix2"])

    def test_invalid_args(self):
        with self.assertRaises(TypeError):
            dist.Backend.register_backend()
        with self.assertRaises(TypeError):
            dist.Backend.register_backend("ut_missing_fn")
        with self.assertRaises(AttributeError):
            dist.Backend.register_backend(12345, _dummy_creator, devices=["cpu"])


if __name__ == "__main__":
    run_tests()
