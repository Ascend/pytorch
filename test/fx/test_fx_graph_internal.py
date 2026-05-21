# Owner(s): ["module: fx"]
"""
Add validation cases for selected torch.fx.graph internal APIs on NPU.

Current covered APIs:
- torch.fx.graph._format_target
- torch.fx.graph._is_from_torch
- torch.fx.graph._origin_type_map.get
- torch.fx.graph._register_custom_builtin

Note:
CodeGen._gen_python_code internally defines a local helper named _format_args.
It is not a torch.fx.graph module-level callable, so it is not added as a
separate test case in this file.
"""

import torch_npu  # noqa: F401

import torch
import torch.fx.graph as fx_graph
from torch.testing._internal.common_utils import TestCase, run_tests


def _test_fx_graph_custom_builtin_for_npu(x):
    return x


class TestFxGraphInternal(TestCase):
    def _remove_custom_builtin(self, name):
        fx_graph._custom_builtins.pop(name, None)
        fx_graph._illegal_names.pop(name, None)

    def test_format_target(self):
        self.assertEqual(
            fx_graph._format_target("root", "foo.bar"),
            "root.foo.bar",
        )
        self.assertEqual(
            fx_graph._format_target("root", "foo.0"),
            'getattr(root.foo, "0")',
        )

    def test_is_from_torch(self):
        self.assertTrue(fx_graph._is_from_torch(torch.add))
        self.assertTrue(fx_graph._is_from_torch(torch.relu))

        def user_defined_func(x):
            return x

        self.assertFalse(fx_graph._is_from_torch(user_defined_func))

    def test_origin_type_map_get(self):
        self.assertEqual(fx_graph._origin_type_map.get(list).__origin__, list)
        self.assertEqual(fx_graph._origin_type_map.get(dict).__origin__, dict)
        self.assertEqual(fx_graph._origin_type_map.get(set).__origin__, set)
        self.assertEqual(fx_graph._origin_type_map.get(tuple).__origin__, tuple)

        self.assertIsNone(fx_graph._origin_type_map.get(TestFxGraphInternal))

    def test_register_custom_builtin(self):
        name = "_test_fx_graph_custom_builtin_for_npu"
        import_str = (
            f"from {__name__} import _test_fx_graph_custom_builtin_for_npu"
        )
        obj = _test_fx_graph_custom_builtin_for_npu

        self._remove_custom_builtin(name)

        fx_graph._register_custom_builtin(name, import_str, obj)

        self.assertIn(name, fx_graph._custom_builtins)
        self.assertEqual(
            fx_graph._custom_builtins[name].import_str,
            import_str,
        )
        self.assertIs(fx_graph._custom_builtins[name].obj, obj)
        self.assertIs(fx_graph._illegal_names[name], obj)

        self._remove_custom_builtin(name)


if __name__ == "__main__":
    run_tests()
