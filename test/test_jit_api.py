# Owner(s): ["oncall: jit"]
"""
Add validation cases for torch.jit APIs:
1. Official jit test files lack sufficient validation for some torch.jit APIs, so this file is added.
2. This file validates:
torch.jit.onednn_fusion_enabled
torch.jit.enable_onednn_fusion
torch.jit.ScriptModule.register_full_backward_hook,
torch.jit.ScriptModule.register_full_backward_pre_hook,
torch.jit.ScriptModule.register_load_state_dict_pre_hook,
torch.jit.ScriptModule.register_load_state_dict_post_hook,
torch.jit.ScriptModule.register_state_dict_pre_hook,
torch.jit.ScriptModule.register_state_dict_post_hook
(extendable)
"""

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestOneDNNJitAPI(TestCase):
    def setUp(self):
        self.original_state = torch.jit.onednn_fusion_enabled()
        super().setUp()

    def tearDown(self):
        torch.jit.enable_onednn_fusion(self.original_state)
        super().tearDown()

    def test_onednn_fusion_enabled_returns_bool(self):
        result = torch.jit.onednn_fusion_enabled()
        self.assertIsInstance(result, bool)

    def test_onednn_fusion_enable_disable_roundtrip(self):
        torch.jit.enable_onednn_fusion(True)
        self.assertEqual(torch.jit.onednn_fusion_enabled(), True)

        torch.jit.enable_onednn_fusion(False)
        self.assertEqual(torch.jit.onednn_fusion_enabled(), False)


class TestScriptModuleHooks(TestCase):
    # register_full_backward_hook: not working for ScriptModule, should raise error

    def test_register_full_backward_hook_raises(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        with self.assertRaises(RuntimeError):
            model.register_full_backward_hook(
                lambda module, grad_input, grad_output: grad_input
            )

    # register_full_backward_pre_hook

    def test_register_full_backward_pre_hook_called(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        called = []
        handle = model.register_full_backward_pre_hook(
            lambda module, grad_output: (called.append(True), grad_output)[1]
        )
        x = torch.randn(2, 2, requires_grad=True).to(device_type)
        model(x).sum().backward()
        self.assertTrue(called)
        handle.remove()

    def test_register_full_backward_pre_hook_modify_grad(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        model.register_full_backward_pre_hook(
            lambda module, grad_output: (grad_output[0] * 0,)
        )
        x = torch.ones(2, device=device_type, requires_grad=True)
        model(x).sum().backward()
        self.assertEqual(x.grad, torch.zeros(2, device=device_type))

    def test_register_full_backward_pre_hook_prepend(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        order = []
        model.register_full_backward_pre_hook(
            lambda module, grad_output: (order.append(1), grad_output)[1]
        )
        model.register_full_backward_pre_hook(
            lambda module, grad_output: (order.append(2), grad_output)[1],
            prepend=True,
        )
        x = torch.randn(2, 2, requires_grad=True).to(device_type)
        model(x).sum().backward()
        self.assertEqual(order, [2, 1])

    def test_register_full_backward_pre_hook_remove(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        called = []
        handle = model.register_full_backward_pre_hook(
            lambda module, grad_output: (called.append(True), grad_output)[1]
        )
        x = torch.randn(2, 2, requires_grad=True).to(device_type)
        model(x).sum().backward()
        self.assertEqual(len(called), 1)
        handle.remove()
        model(x).sum().backward()
        self.assertEqual(len(called), 1)

    # register_load_state_dict_pre_hook & register_load_state_dict_post_hook

    def test_load_state_dict_pre_hook_fires_before_module_and_post_hook(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        order = []

        def post_hook(module, incompatible_keys):
            order.append("post")

        def pre_hook(
            module,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            order.append("pre")

        model = M().to(device_type)
        model.register_load_state_dict_post_hook(post_hook)
        model.register_load_state_dict_pre_hook(pre_hook)
        model.load_state_dict(model.state_dict())
        self.assertEqual(order, ["pre", "post"])

        order2 = []

        def pre_hook_modify(
            module,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            order2.append("pre")
            for key in list(state_dict.keys()):
                state_dict[key] = torch.zeros_like(state_dict[key]).to(device_type)

        def post_hook_check(module, incompatible_keys):
            order2.append("post")
            self.assertEqual(
                module.linear.weight,
                torch.zeros_like(module.linear.weight).to(device_type),
            )

        model2 = M().to(device_type)
        model2.register_load_state_dict_post_hook(post_hook_check)
        model2.register_load_state_dict_pre_hook(pre_hook_modify)
        model2.load_state_dict(model.state_dict())
        self.assertEqual(order2, ["pre", "post"])

    def test_register_load_state_dict_pre_hook_called(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        called = []

        def hook(
            module,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            called.append(prefix)

        model.register_load_state_dict_pre_hook(hook)
        model.load_state_dict(model.state_dict())
        self.assertEqual(called, [""])

    def test_register_load_state_dict_pre_hook_with_module(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        received_module = []

        def hook(
            module,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            received_module.append(module)

        model.register_load_state_dict_pre_hook(hook)
        model.load_state_dict(model.state_dict())
        self.assertIs(received_module[0], model)

    def test_register_load_state_dict_pre_hook_remove(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        called = []

        def hook(
            module,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            called.append(True)

        handle = model.register_load_state_dict_pre_hook(hook)
        model.load_state_dict(model.state_dict())
        self.assertEqual(len(called), 1)
        handle.remove()
        model.load_state_dict(model.state_dict())
        self.assertEqual(len(called), 1)

    # register_load_state_dict_post_hook

    def test_register_load_state_dict_post_hook_called(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        called = []

        def hook(module, incompatible_keys):
            called.append(True)

        handle = model.register_load_state_dict_post_hook(hook)
        model.load_state_dict(model.state_dict())
        self.assertTrue(called)
        handle.remove()

    def test_register_load_state_dict_post_hook_with_module(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        received_module = []

        def hook(module, incompatible_keys):
            received_module.append(module)

        handle = model.register_load_state_dict_post_hook(hook)
        model.load_state_dict(model.state_dict())
        self.assertIs(received_module[0], model)
        handle.remove()

    def test_register_load_state_dict_post_hook_remove(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        called = []

        def hook(module, incompatible_keys):
            called.append(True)

        handle = model.register_load_state_dict_post_hook(hook)
        model.load_state_dict(model.state_dict())
        self.assertEqual(len(called), 1)
        handle.remove()
        model.load_state_dict(model.state_dict())
        self.assertEqual(len(called), 1)

    # register_state_dict_pre_hook & register_state_dict_post_hook

    def test_state_dict_pre_hook_fires_before_module_and_post_hook(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        order = []

        def post_hook(module, state_dict, prefix, local_metadata):
            order.append("post")

        def pre_hook(module, prefix, keep_vars):
            order.append("pre")

        model = M().to(device_type)
        model.register_state_dict_post_hook(post_hook)
        model.register_state_dict_pre_hook(pre_hook)
        model.state_dict()
        self.assertEqual(order, ["pre", "post"])

        order2 = []
        pre_hook_called = [False]

        def pre_hook_flag(module, prefix, keep_vars):
            order2.append("pre")
            pre_hook_called[0] = True

        def post_hook_check(module, state_dict, prefix, local_metadata):
            order2.append("post")
            self.assertTrue(pre_hook_called[0])
            self.assertTrue(len(state_dict) > 0)

        model2 = M().to(device_type)
        model2.register_state_dict_post_hook(post_hook_check)
        model2.register_state_dict_pre_hook(pre_hook_flag)
        model2.state_dict()
        self.assertEqual(order2, ["pre", "post"])

    def test_register_state_dict_pre_hook_called(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        called = []

        def hook(module, prefix, keep_vars):
            called.append(prefix)

        model.register_state_dict_pre_hook(hook)
        model.state_dict()
        self.assertEqual(called, [""])

    def test_register_state_dict_pre_hook_with_module(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        received_module = []

        def hook(module, prefix, keep_vars):
            received_module.append(module)

        model.register_state_dict_pre_hook(hook)
        model.state_dict()
        self.assertIs(received_module[0], model)

    def test_register_state_dict_pre_hook_remove(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        called = []

        def hook(module, prefix, keep_vars):
            called.append(True)

        handle = model.register_state_dict_pre_hook(hook)
        model.state_dict()
        self.assertEqual(len(called), 1)
        handle.remove()
        model.state_dict()
        self.assertEqual(len(called), 1)

    # register_state_dict_post_hook

    def test_register_state_dict_post_hook_called(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        called = []

        def hook(module, state_dict, prefix, local_metadata):
            called.append(prefix)

        model.register_state_dict_post_hook(hook)
        model.state_dict()
        self.assertEqual(called, [""])

    def test_register_state_dict_post_hook_with_module(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        received_module = []

        def hook(module, state_dict, prefix, local_metadata):
            received_module.append(module)

        model.register_state_dict_post_hook(hook)
        model.state_dict()
        self.assertIs(received_module[0], model)

    def test_register_state_dict_post_hook_remove(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        called = []

        def hook(module, state_dict, prefix, local_metadata):
            called.append(True)

        handle = model.register_state_dict_post_hook(hook)
        model.state_dict()
        self.assertEqual(len(called), 1)
        handle.remove()
        model.state_dict()
        self.assertEqual(len(called), 1)


if __name__ == "__main__":
    run_tests()
