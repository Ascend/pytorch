# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.container.ModuleDict.keys 接口功能正确性
API 名称：torch.nn.modules.container.ModuleDict.keys
API 签名：keys(self) -> container_abcs.KeysView[str]

覆盖维度表：
| 覆盖维度         | 说明                                        | 覆盖情况                            |
|------------------|---------------------------------------------|-------------------------------------|
| 空/非空          | 空 ModuleDict 的 keys 为空视图；非空时逐键返回 | 已覆盖（keys_empty_dict/keys_insertion_order_and_content） |
| 枚举选项         | 无枚举参数                                  | 未覆盖（无该类参数）                |
| 参数类型         | 返回 dict_keys（KeysView）；成员为 str 键名 | 已覆盖（keys_view_type_and_membership） |
| 传参与不传参     | 无参方法                                    | 已覆盖（全部用例无参调用）          |
| 等价类/边界值     | 空字典/单键/多键；删除键后视图同步收缩      | 已覆盖（keys_empty_dict/keys_reflects_deletion） |
| 正常传参场景     | 键序与插入序一致，且与 _modules 注册序一致；视图为活视图，随增删动态变化 | 已覆盖（keys_insertion_order_and_content/keys_live_view） |
| 异常传参场景     | 无参方法，无异常入参路径                    | 未覆盖（无参方法，无异常入参路径）  |
| 混合设备类型     | ModuleDict 持有 NPU 驻留子模块时 keys 与设备无关，按键可正常调用子模块前向 | 已覆盖（keys_with_device_modules）  |

未覆盖项及原因：
- 无。keys 为纯键集视图方法，无设备计算路径；设备驻留子模块场景在
  keys_with_device_modules 中覆盖（NPU 环境实测，CUDA 环境同样可运行）。
"""
import os
import unittest
from collections.abc import KeysView

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化，避免
# 不必要的探测开销；被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
DEVICE_NAME = 'cpu'
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = 'npu'
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestModuleDictkeys(TestCase):
    """Test cases for torch.nn.modules.container.ModuleDict.keys (NPU/CUDA).

    ModuleDict.keys is ``self._modules.keys()`` behind @_copy_to_script_wrapper
    (torch 2.15.0, torch/nn/modules/container.py). Empirical contract
    (torch 2.15.0+cpu + torch_npu 2.15.0.dev20261118, probed on Ascend910B4):
      - an empty ModuleDict yields an empty dict_keys view;
      - the view is a live dict view: later insertions and deletions are
        reflected in previously obtained views;
      - key order follows insertion order and matches the _modules order;
      - membership works via the ``in`` operator on the view;
      - keys are plain strings, independent of the child modules' devices.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_keys_empty_dict(self):
        """An empty ModuleDict yields an empty keys view."""
        md = torch.nn.ModuleDict()
        keys = md.keys()
        self.assertEqual(len(keys), 0, "empty ModuleDict must have no keys")
        self.assertEqual(list(keys), [], "keys() must iterate to an empty list")

    def test_keys_view_type_and_membership(self):
        """keys() returns a KeysView whose membership tracks registered names."""
        md = torch.nn.ModuleDict({'linear': torch.nn.Linear(2, 2)})
        keys = md.keys()
        self.assertIsInstance(keys, KeysView,
                              "keys() must return a KeysView")
        self.assertIn('linear', keys, "registered keys must be members")
        self.assertNotIn('conv', keys, "unregistered names must not be members")
        # The ``in`` contract on the view equals the one on the ModuleDict.
        self.assertEqual('linear' in keys, 'linear' in md)

    def test_keys_insertion_order_and_content(self):
        """Key order follows insertion order and matches _modules order."""
        md = torch.nn.ModuleDict()
        md['linear'] = torch.nn.Linear(2, 2)
        md['conv'] = torch.nn.Conv2d(1, 1, 1)
        md['act'] = torch.nn.ReLU()
        self.assertEqual(list(md.keys()), ['linear', 'conv', 'act'],
                         "keys must preserve insertion order")
        self.assertEqual(list(md.keys()), list(md._modules.keys()),
                         "keys() must agree with the _modules registration order")

    def test_keys_live_view(self):
        """A previously obtained view reflects later insertions and deletions."""
        md = torch.nn.ModuleDict({'linear': torch.nn.Linear(2, 2)})
        view = md.keys()
        md['act'] = torch.nn.ReLU()
        self.assertIn('act', view,
                      "the live view must reflect later insertions")
        del md['linear']
        self.assertNotIn('linear', view,
                         "the live view must reflect later deletions")
        self.assertEqual(list(view), ['act'])

    def test_keys_reflects_deletion(self):
        """keys() shrinks after pop/del and stays consistent with len()."""
        md = torch.nn.ModuleDict({'a': torch.nn.ReLU(), 'b': torch.nn.ReLU()})
        self.assertEqual(len(md.keys()), 2)
        md.pop('a')
        self.assertEqual(list(md.keys()), ['b'],
                         "popped keys must disappear from keys()")
        self.assertEqual(len(md.keys()), 1)

    def test_keys_with_device_modules(self):
        """Keys are device-independent; child modules stay callable by key."""
        md = torch.nn.ModuleDict({
            'dense': torch.nn.Linear(2, 2).to(self.device),
            'act': torch.nn.ReLU(),
        })
        keys = md.keys()
        self.assertEqual(sorted(keys), ['act', 'dense'],
                         "keys must be plain strings regardless of child devices")
        # Iterating by key must reach working (device-resident) modules.
        y = md['dense'](torch.ones(1, 2, device=self.device))
        self.assertEqual(y.device.type, self.device_name,
                         "the child module must run on its own device")
        torch.testing.assert_close(
            md['act'](y).cpu(), torch.relu(y.detach().cpu()),
            msg="the ReLU child must apply its documented semantics by key")


if __name__ == "__main__":
    run_tests()
