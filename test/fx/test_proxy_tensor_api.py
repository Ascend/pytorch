# Owner(s): ["module: fx"]

"""
Add validation cases for torch.fx.experimental.proxy_tensor APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for
   some proxy_tensor APIs, so this file is added.
2. This file validates get_proxy_mode, handle_sym_dispatch, make_fx,
   maybe_enable_thunkify, and maybe_disable_thunkify (extendable).
"""

import torch
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import (
    get_proxy_mode,
    handle_sym_dispatch,
    make_fx,
    maybe_disable_thunkify,
    maybe_enable_thunkify,
)
from torch.testing._internal.common_utils import run_tests, TestCase


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestProxyTensorAPI(TestCase):
    def test_make_fx_returns_graph_module(self):
        def fn(x, y):
            return torch.relu(x + y)

        x = torch.randn(2, 3).to(device_type)
        y = torch.randn(2, 3).to(device_type)

        gm = make_fx(fn)(x, y)

        self.assertIsInstance(gm, GraphModule)
        self.assertEqual(gm(x, y), fn(x, y))
        self.assertIn("aten", str(gm.graph))

    def test_get_proxy_mode_during_make_fx(self):
        modes = []

        def fn(x):
            modes.append(get_proxy_mode())
            return torch.sin(x) + 1

        self.assertIsNone(get_proxy_mode())

        x = torch.randn(2, 3).to(device_type)
        gm = make_fx(fn)(x)

        self.assertIsInstance(gm, GraphModule)
        self.assertEqual(gm(x), torch.sin(x) + 1)
        self.assertGreaterEqual(len(modes), 1)
        self.assertIsNotNone(modes[0])

    def test_handle_sym_dispatch_requires_proxy_mode(self):
        def fn(x):
            return x

        self.assertTrue(callable(handle_sym_dispatch))
        self.assertIsNone(get_proxy_mode())

        with self.assertRaises(AssertionError):
            handle_sym_dispatch(fn, (3,), {})

    def test_thunkify_context_managers(self):
        def fn(x):
            with maybe_enable_thunkify():
                y = x + 1
                with maybe_disable_thunkify():
                    z = y * 2
                return z

        x = torch.randn(2, 3).to(device_type)
        gm = make_fx(fn)(x)

        self.assertIsInstance(gm, GraphModule)
        self.assertEqual(gm(x), fn(x))


if __name__ == "__main__":
    run_tests()
