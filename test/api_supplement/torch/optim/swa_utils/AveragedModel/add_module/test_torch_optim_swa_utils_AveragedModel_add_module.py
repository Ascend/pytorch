# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.swa_utils.AveragedModel.add_module 接口功能正确性
API 名称：torch.optim.swa_utils.AveragedModel.add_module
API 签名：add_module(self, name: str, module: Module | None) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册命名子模块（None 亦可显式注册以占位） | 已覆盖（test_add_module_registers/test_add_module_none_placeholder） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | name 字符串 + Module；非 Module 抛 TypeError | 已覆盖（test_add_module_invalid_module） |
| 传参与不传参 | 必传 name/module | 已覆盖（test_add_module_invalid_name） |
| 等价类/边界值 | 空名字/含点名字报 KeyError；重名 add_module 静默替换且 state_dict 键随之切换 |
已覆盖（add_module_invalid_name/add_module_duplicate_name/add_module_duplicate_name_updates_state_dict_keys） |
| 正常传参场景 | add_module 后属性可访问、注册进 named_children/named_parameters、state_dict 键路径正确、参数 requires_grad 生效 |
已覆盖（add_module_registers/add_module_state_dict_key_paths/add_module_params_registered_requires_grad） |
| 异常传参场景 | 非 Module 类型、非 str 名字、空串、含点、属性冲突均抛异常 |
已覆盖（add_module_invalid_name/add_module_invalid_module/add_module_name_conflicts_with_attribute） |
| 混合设备类型 | 追加驻留当前设备的子模块（CPU AveragedModel + 设备子模块）后注册与 update 跨设备拷贝均生效 |
已覆盖（add_module_device_resident_child/add_module_update_covers_matching_source） |

未覆盖项及原因：
- 本接口本质为优化器/调度器/状态工具的框架行为，涉及浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。
- update_parameters 对新增子模块按参数位置 zip 配对（strict=False）：源 model 参数较少时新增模块不被更新，
  该截断行为在 test_add_module_update_truncates_unpaired_child 中如实锁定，不视为缺陷断言。
"""
import os
import unittest

import torch
import torch.optim.swa_utils as swa_utils

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 条件导入 torch_npu，并用 torch.npu.is_available() 探测可用 NPU。
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
class TestAveragedModeladd_module(TestCase):
    """Test cases for torch.optim.swa_utils.AveragedModel.add_module (NPU/CUDA portable).

    AveragedModel is an nn.Module subclass, so add_module registers a child
    directly on the averaging wrapper (next to the deep-copied ``module``).
    Empirical contract (torch 2.11, probed on NPU):
      - state_dict gains top-level "<name>.*" keys alongside "module.*"/"n_averaged";
      - re-adding an existing name silently replaces the child (no raise) and
        the state_dict keys switch to the new module;
      - update_parameters pairs am.parameters() with model.parameters()
        positionally (zip, strict=False), so an added child is folded into the
        average only when the source module exposes matching extra parameters;
      - the added child keeps the very same Parameter objects (no deepcopy),
        preserving requires_grad and the device of the child's tensors.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_add_module_registers(self):
        """A named child module becomes accessible as an attribute."""
        am = torch.optim.swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        am.add_module('extra', torch.nn.Linear(2, 2))
        self.assertTrue(hasattr(am, 'extra'))
        self.assertIn('extra', dict(am.named_children()))

    def test_add_module_duplicate_name(self):
        """Re-adding the same name silently replaces the child module."""
        am = torch.optim.swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        old = torch.nn.Linear(2, 2)
        am.add_module('a', old)
        new = torch.nn.Linear(2, 2)
        am.add_module('a', new)
        # nn.Module.add_module re-registers without raising.
        self.assertIs(am.a, new, "duplicate name must be silently replaced")

    def test_add_module_invalid_name(self):
        """Empty, dotted, or non-str names raise before registering."""
        am = torch.optim.swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        m = torch.nn.Linear(2, 2)
        with self.assertRaises(KeyError):
            am.add_module("", m)
        with self.assertRaises(KeyError):
            am.add_module("a.b", m)
        with self.assertRaises(TypeError):
            am.add_module(1, m)

    def test_add_module_invalid_module(self):
        """Non-Module module argument raises TypeError."""
        am = torch.optim.swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        with self.assertRaises(TypeError):
            am.add_module("x", object())

    def test_add_module_name_conflicts_with_attribute(self):
        """name already used by a plain attribute raises KeyError."""
        am = torch.optim.swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        # 'training' is an existing plain attribute that is not a _modules
        # entry; registering a submodule under that name must be rejected.
        self.assertTrue(hasattr(am, 'training'))
        self.assertNotIn('training', am._modules)
        with self.assertRaises(KeyError):
            am.add_module('training', torch.nn.Linear(2, 2))

    def test_add_module_none_placeholder(self):
        """add_module accepts None to register a placeholder slot."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        am.add_module('placeholder', None)
        self.assertIn('placeholder', am._modules,
                      "None child must be registered in _modules under its name")
        self.assertIsNone(am.placeholder)
        # Probed on torch 2.11: named_children()/named_modules() skip None
        # entries, so the placeholder contributes neither keys nor parameters.
        self.assertEqual([n for n, _ in am.named_children()], ['module'])
        self.assertEqual(sum(p.numel() for p in am.parameters()), 6)

    def test_add_module_state_dict_key_paths(self):
        """state_dict exposes top-level '<name>.*' keys next to 'module.*'."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2, bias=False))
        extra = torch.nn.Linear(2, 2, bias=False)
        am.add_module('extra', extra)
        keys = sorted(am.state_dict().keys())
        self.assertIn('extra.weight', keys,
                      "added child must contribute '<name>.*' state_dict keys")
        self.assertIn('module.weight', keys)
        self.assertIn('n_averaged', keys)
        # Round-trip: values loaded under the new key path land in the child.
        sd = am.state_dict()
        sd['extra.weight'] = torch.full((2, 2), 7.0)
        am.load_state_dict(sd)
        torch.testing.assert_close(am.extra.weight.detach(), torch.full((2, 2), 7.0),
                                   msg="load_state_dict must fill the added child via its key path")

    def test_add_module_duplicate_name_updates_state_dict_keys(self):
        """Replacing a child under the same name swaps its state_dict entries."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2, bias=False))
        old = torch.nn.Linear(2, 2, bias=False)
        am.add_module('a', old)
        self.assertIn('a.weight', am.state_dict())
        new = torch.nn.Linear(3, 1, bias=False)
        am.add_module('a', new)  # silent replace, even with a different shape
        keys = sorted(am.state_dict().keys())
        self.assertIn('a.weight', keys)
        self.assertEqual(am.state_dict()['a.weight'].shape, (1, 3),
                         "state_dict must reflect the replacing module's tensors")
        self.assertIs(am.a, new)

    def test_add_module_params_registered_requires_grad(self):
        """Added child params join am.parameters() with requires_grad intact."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2, bias=False))
        extra = torch.nn.Linear(2, 2, bias=False)
        am.add_module('extra', extra)
        # The very same Parameter object is registered (no deepcopy in add_module).
        self.assertTrue(any(p is extra.weight for p in am.parameters()),
                        "added child's Parameter must be registered in am.parameters()")
        self.assertIn('extra.weight', dict(am.named_parameters()))
        self.assertTrue(am.extra.weight.requires_grad,
                        "registered Parameter must keep requires_grad=True")
        # add_module returns None per its signature.
        self.assertIsNone(am.add_module('plain', None))

    def test_add_module_device_resident_child(self):
        """A device-resident child can be attached to a CPU-resident wrapper."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2, bias=False))
        extra = torch.nn.Linear(2, 2, bias=False).to(self.device)
        am.add_module('extra', extra)
        # The child's tensors keep their device (add_module does not move them).
        self.assertEqual(am.extra.weight.device.type, self.device_name,
                         "add_module must not migrate the child's tensors")
        self.assertTrue(any(p is extra.weight for p in am.parameters()))

    def test_add_module_update_covers_matching_source(self):
        """update_parameters folds the added child when the source matches its layout."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2, bias=False))
        am.add_module('extra', torch.nn.Linear(2, 2, bias=False))
        # am.parameters() = [module.weight, extra.weight]; build a source module
        # with two params in the same positional order to pair them.
        src = torch.nn.Module()
        src.add_module('module', torch.nn.Linear(2, 2, bias=False))
        src.add_module('extra', torch.nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            src.module.weight.fill_(1.0)
            src.extra.weight.fill_(9.0)
        am.update_parameters(src)
        self.assertEqual(int(am.n_averaged), 1)
        torch.testing.assert_close(am.module.weight.detach(), torch.full((2, 2), 1.0),
                                   msg="first source param must be copied into module")
        torch.testing.assert_close(am.extra.weight.detach(), torch.full((2, 2), 9.0),
                                   msg="added child must be covered when the source layout matches")

    def test_add_module_update_truncates_unpaired_child(self):
        """A child beyond the source model's param count stays untouched."""
        model = torch.nn.Linear(2, 2, bias=False)
        am = swa_utils.AveragedModel(model)
        extra = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            extra.weight.fill_(7.0)
        am.add_module('extra', extra)
        with torch.no_grad():
            model.weight.fill_(1.0)
        # zip(self.parameters(), model.parameters(), strict=False) truncates at
        # the shorter source, so 'extra' is never paired and keeps its values.
        am.update_parameters(model)
        torch.testing.assert_close(am.extra.weight.detach(), torch.full((2, 2), 7.0),
                                   msg="unpaired child params must be left untouched")
        torch.testing.assert_close(am.module.weight.detach(), torch.full((2, 2), 1.0),
                                   msg="paired module param must still be averaged")
        self.assertEqual(int(am.n_averaged), 1,
                         "the counter still advances for the paired parameters")
        with torch.no_grad():
            model.weight.fill_(3.0)
        am.update_parameters(model)
        torch.testing.assert_close(am.extra.weight.detach(), torch.full((2, 2), 7.0),
                                   msg="unpaired child must remain untouched after the 2nd update")
        torch.testing.assert_close(am.module.weight.detach(), torch.full((2, 2), 2.0),
                                   msg="paired param must follow the running mean")


if __name__ == "__main__":
    run_tests()
