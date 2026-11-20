# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.container.ParameterDict.keys 接口功能正确性
API 名称：torch.nn.modules.container.ParameterDict.keys
API 签名：keys(self) -> container_abcs.KeysView[str]

覆盖维度表：
| 覆盖维度         | 说明                                        | 覆盖情况                            |
|------------------|---------------------------------------------|-------------------------------------|
| 空/非空          | 空 ParameterDict 的 keys 为空视图；非空时逐键返回 | 已覆盖（keys_empty_dict/keys_insertion_order_and_content） |
| 枚举选项         | 无枚举参数                                  | 未覆盖（无该类参数）                |
| 参数类型         | 返回 dict_keys（KeysView）；成员为 str 键名 | 已覆盖（keys_view_type_and_membership） |
| 传参与不传参     | 无参方法                                    | 已覆盖（全部用例无参调用）          |
| 等价类/边界值     | 空字典/单键/多键；删除键后视图同步收缩      | 已覆盖（keys_empty_dict/keys_reflects_deletion） |
| 正常传参场景     | OrderedDict/ParameterDict 输入保持原序；普通 Mapping 输入经 update 排序；
|                  | 视图与 _keys 一致且为活视图 | 已覆盖（keys_insertion_order_and_content
|                  | /keys_update_ordering/keys_live_view） |
| 异常传参场景     | 无参方法，无异常入参路径                    | 未覆盖（无参方法，无异常入参路径）  |
| 混合设备类型     | keys 与参数设备无关；NPU 驻留参数注册后键集不变 | 已覆盖（keys_with_device_params）   |

未覆盖项及原因：
- 无。keys 为纯键集视图方法，无设备计算路径；设备驻留参数场景在
  keys_with_device_params 中覆盖（NPU 环境实测，CUDA 环境同样可运行）。
"""
import os
import unittest
from collections import OrderedDict
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
class TestParameterDictkeys(TestCase):
    """Test cases for torch.nn.modules.container.ParameterDict.keys (NPU/CUDA).

    ParameterDict.keys is ``self._keys.keys()`` (torch 2.15.0,
    torch/nn/modules/container.py). Empirical contract (torch 2.15.0+cpu +
    torch_npu 2.15.0.dev20261118, probed on Ascend910B4):
      - an empty ParameterDict yields an empty dict_keys view;
      - the view is a live dict view over _keys: later insertions and
        deletions are reflected in previously obtained views;
      - keys from OrderedDict/ParameterDict inputs keep their original order;
      - keys from a plain Mapping input via update() come back sorted;
      - membership works via ``in`` and equals the ParameterDict membership;
      - keys are plain strings, independent of the parameters' devices.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_keys_empty_dict(self):
        """An empty ParameterDict yields an empty keys view."""
        pd = torch.nn.ParameterDict()
        keys = pd.keys()
        self.assertEqual(len(keys), 0, "empty ParameterDict must have no keys")
        self.assertEqual(list(keys), [], "keys() must iterate to an empty list")

    def test_keys_view_type_and_membership(self):
        """keys() returns a KeysView whose membership tracks registered names."""
        pd = torch.nn.ParameterDict({'left': torch.nn.Parameter(torch.randn(2))})
        keys = pd.keys()
        self.assertIsInstance(keys, KeysView,
                              "keys() must return a KeysView")
        self.assertIn('left', keys, "registered keys must be members")
        self.assertNotIn('nope', keys, "unregistered names must not be members")
        self.assertEqual('left' in keys, 'left' in pd,
                         "view membership must equal the ParameterDict membership")

    def test_keys_insertion_order_and_content(self):
        """OrderedDict/sequence inputs keep their order and match _keys."""
        ordered = OrderedDict([('z', torch.nn.Parameter(torch.randn(1))),
                               ('y', torch.nn.Parameter(torch.randn(1))),
                               ('x', torch.nn.Parameter(torch.randn(1)))])
        pd = torch.nn.ParameterDict(ordered)
        self.assertEqual(list(pd.keys()), ['z', 'y', 'x'],
                         "OrderedDict init must preserve its key order")
        self.assertEqual(list(pd.keys()), list(pd._keys.keys()),
                         "keys() must agree with the _keys registration order")

    def test_keys_update_ordering(self):
        """update() from a plain Mapping sorts; from a ParameterDict it preserves."""
        pd = torch.nn.ParameterDict()
        pd.update({'b': torch.nn.Parameter(torch.randn(1)),
                   'a': torch.nn.Parameter(torch.randn(1))})
        self.assertEqual(list(pd.keys()), ['a', 'b'],
                         "plain-Mapping update must yield sorted keys")
        other = torch.nn.ParameterDict(OrderedDict([
            ('zz', torch.nn.Parameter(torch.randn(1))),
            ('mm', torch.nn.Parameter(torch.randn(1))),
        ]))
        pd.update(other)
        self.assertEqual(list(pd.keys()), ['a', 'b', 'zz', 'mm'],
                         "ParameterDict update must preserve the source order")

    def test_keys_live_view(self):
        """A previously obtained view reflects later insertions and deletions."""
        pd = torch.nn.ParameterDict({'left': torch.nn.Parameter(torch.randn(2))})
        view = pd.keys()
        pd['mid'] = torch.nn.Parameter(torch.randn(1))
        self.assertIn('mid', view,
                      "the live view must reflect later insertions")
        del pd['left']
        self.assertNotIn('left', view,
                         "the live view must reflect later deletions")
        self.assertEqual(list(view), ['mid'])

    def test_keys_reflects_deletion(self):
        """keys() shrinks after pop/del and stays consistent with len()."""
        pd = torch.nn.ParameterDict({'a': torch.nn.Parameter(torch.randn(1)),
                                     'b': torch.nn.Parameter(torch.randn(1))})
        self.assertEqual(len(pd.keys()), 2)
        pd.pop('a')
        self.assertEqual(list(pd.keys()), ['b'],
                         "popped keys must disappear from keys()")
        self.assertEqual(len(pd.keys()), 1)

    def test_keys_with_device_params(self):
        """Keys are device-independent; device-resident params keep their keys."""
        pd = torch.nn.ParameterDict({
            'dense': torch.nn.Parameter(torch.randn(2, device=self.device)),
            'bias': torch.nn.Parameter(torch.zeros(2, device=self.device)),
        })
        self.assertEqual(sorted(pd.keys()), ['bias', 'dense'],
                         "keys must be plain strings regardless of param devices")
        self.assertEqual(pd['dense'].device.type, self.device_name,
                         "the registered parameter must keep its device")
        torch.testing.assert_close(
            pd['bias'].detach().cpu(), torch.zeros(2),
            msg="the registered parameter value must be intact")


if __name__ == "__main__":
    run_tests()
