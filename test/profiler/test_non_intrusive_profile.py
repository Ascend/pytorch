from unittest.mock import patch

import torch

import torch_npu.profiler._non_intrusive_profile as none_intrusive_profile
from torch_npu.profiler._non_intrusive_profile import _NonIntrusiveProfile
from torch_npu.testing.testcase import TestCase, run_tests


class TestNoneInstrusiveProfile(TestCase):
    def setUp(self):
        super().setUp()
        self._cleanup_optimizer_step_hook()
        self.dynamic_profiler_model = (
            none_intrusive_profile.DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL
        )

    def tearDown(self):
        self._cleanup_optimizer_step_hook()
        none_intrusive_profile.DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL = (
            self.dynamic_profiler_model
        )
        super().tearDown()

    @staticmethod
    def _cleanup_optimizer_step_hook():
        handle = _NonIntrusiveProfile._optimizer_step_hook_handle
        if handle is not None:
            handle.remove()
            _NonIntrusiveProfile._optimizer_step_hook_handle = None
        _NonIntrusiveProfile._optimizer_steps.clear()
        _NonIntrusiveProfile._profiler_step = 0

    @staticmethod
    def _getenv(values):
        return lambda name, default=None: values.get(name, default)

    def test_register_optimizer_step_hook_is_idempotent(self):
        _NonIntrusiveProfile._register_optimizer_step_hook()
        first_handle = _NonIntrusiveProfile._optimizer_step_hook_handle

        _NonIntrusiveProfile._register_optimizer_step_hook()

        self.assertIs(first_handle, _NonIntrusiveProfile._optimizer_step_hook_handle)

    def test_global_hook_aggregates_different_optimizers_by_max(self):
        optimizer1 = torch.optim.SGD([torch.tensor([1.0])], lr=0.01)
        optimizer2 = torch.optim.Adam([torch.tensor([1.0])], lr=0.01)

        with patch.object(none_intrusive_profile, "dp_step") as mock_dp_step:
            _NonIntrusiveProfile._register_optimizer_step_hook()
            optimizer1.step()
            optimizer2.step()

        mock_dp_step.assert_called_once_with()

    def test_global_hook_aggregates_optimizers_of_the_same_class(self):
        optimizer1 = torch.optim.SGD([torch.tensor([1.0])], lr=0.01)
        optimizer2 = torch.optim.SGD([torch.tensor([2.0])], lr=0.01)

        with patch.object(none_intrusive_profile, "dp_step") as mock_dp_step:
            _NonIntrusiveProfile._register_optimizer_step_hook()
            optimizer1.step()
            optimizer2.step()

        mock_dp_step.assert_called_once_with()

    def test_global_hook_uses_max_for_uneven_optimizer_frequencies(self):
        optimizer1 = torch.optim.SGD([torch.tensor([1.0])], lr=0.01)
        optimizer2 = torch.optim.Adam([torch.tensor([1.0])], lr=0.01)

        with patch.object(none_intrusive_profile, "dp_step") as mock_dp_step:
            _NonIntrusiveProfile._register_optimizer_step_hook()
            for iteration in range(6):
                optimizer1.step()
                if (iteration + 1) % 3 == 0:
                    optimizer2.step()

        self.assertEqual(mock_dp_step.call_count, 6)

    def test_global_hook_advances_for_each_single_optimizer_step(self):
        optimizer = torch.optim.SGD([torch.tensor([1.0])], lr=0.01)

        with patch.object(none_intrusive_profile, "dp_step") as mock_dp_step:
            _NonIntrusiveProfile._register_optimizer_step_hook()
            for _ in range(3):
                optimizer.step()

        self.assertEqual(mock_dp_step.call_count, 3)

    def test_hook_applies_to_optimizer_created_before_registration(self):
        optimizer = torch.optim.SGD([torch.tensor([1.0])], lr=0.01)

        with patch.object(none_intrusive_profile, "dp_step") as mock_dp_step:
            _NonIntrusiveProfile._register_optimizer_step_hook()
            optimizer.step()

        mock_dp_step.assert_called_once_with()

    def test_hook_applies_to_optimizer_created_after_registration(self):
        with patch.object(none_intrusive_profile, "dp_step") as mock_dp_step:
            _NonIntrusiveProfile._register_optimizer_step_hook()
            optimizer = torch.optim.SGD([torch.tensor([1.0])], lr=0.01)
            optimizer.step()

        mock_dp_step.assert_called_once_with()

    def test_step_without_grad_still_advances_profiler(self):
        parameter = torch.tensor([1.0], requires_grad=True)
        optimizer = torch.optim.SGD([parameter], lr=0.01)

        with patch.object(none_intrusive_profile, "dp_step") as mock_dp_step:
            _NonIntrusiveProfile._register_optimizer_step_hook()
            optimizer.step()

        mock_dp_step.assert_called_once_with()

    def test_gradient_accumulation_advances_only_on_optimizer_step(self):
        parameter = torch.tensor([1.0], requires_grad=True)
        optimizer = torch.optim.SGD([parameter], lr=0.01)

        with patch.object(none_intrusive_profile, "dp_step") as mock_dp_step:
            _NonIntrusiveProfile._register_optimizer_step_hook()
            for _ in range(4):
                (parameter * 2).sum().backward()
            optimizer.step()

        mock_dp_step.assert_called_once_with()

    def test_hook_is_not_called_when_optimizer_step_raises(self):
        class FailingOptimizer(torch.optim.Optimizer):
            def __init__(self, params):
                super().__init__(params, {})

            def step(self, closure=None):
                raise RuntimeError("optimizer step failed")

        optimizer = FailingOptimizer([torch.tensor([1.0])])

        with patch.object(none_intrusive_profile, "dp_step") as mock_dp_step:
            _NonIntrusiveProfile._register_optimizer_step_hook()
            with self.assertRaisesRegex(RuntimeError, "optimizer step failed"):
                optimizer.step()

        mock_dp_step.assert_not_called()

    def test_instance_hook_runs_before_global_hook(self):
        events = []
        optimizer = torch.optim.SGD([torch.tensor([1.0])], lr=0.01)
        instance_handle = optimizer.register_step_post_hook(
            lambda _optimizer, _args, _kwargs: events.append("instance")
        )
        self.addCleanup(instance_handle.remove)

        with patch.object(none_intrusive_profile, "dp_step", side_effect=lambda: events.append("global")):
            _NonIntrusiveProfile._register_optimizer_step_hook()
            optimizer.step()

        self.assertEqual(events, ["instance", "global"])

    def test_init_does_not_register_hook_when_profiling_is_disabled(self):
        env = {
            "PROF_CONFIG_PATH": "",
            "KINETO_USE_DAEMON": None,
            "MSMONITOR_USE_DAEMON": None,
        }
        with patch.object(none_intrusive_profile.os, "getenv", side_effect=self._getenv(env)), \
             patch.object(none_intrusive_profile, "dp_init") as mock_dp_init, \
             patch.object(_NonIntrusiveProfile, "_register_optimizer_step_hook") as mock_register:
            _NonIntrusiveProfile.init()

        mock_dp_init.assert_not_called()
        mock_register.assert_not_called()

    def test_msmonitor_trigger_registers_hook(self):
        env = {
            "PROF_CONFIG_PATH": "",
            "KINETO_USE_DAEMON": None,
            "MSMONITOR_USE_DAEMON": "1",
        }
        with patch.object(none_intrusive_profile.os, "getenv", side_effect=self._getenv(env)), \
             patch.object(none_intrusive_profile, "dp_init") as mock_dp_init, \
             patch.object(_NonIntrusiveProfile, "_register_optimizer_step_hook") as mock_register:
            _NonIntrusiveProfile.init()

        mock_dp_init.assert_called_once_with("")
        mock_register.assert_called_once_with()

    def test_kineto_trigger_registers_hook(self):
        env = {
            "PROF_CONFIG_PATH": "",
            "KINETO_USE_DAEMON": "1",
            "MSMONITOR_USE_DAEMON": None,
        }
        with patch.object(none_intrusive_profile.os, "getenv", side_effect=self._getenv(env)), \
             patch.object(none_intrusive_profile, "dp_init") as mock_dp_init, \
             patch.object(none_intrusive_profile, "print_warn_msg") as mock_print_warn, \
             patch.object(_NonIntrusiveProfile, "_register_optimizer_step_hook") as mock_register:
            _NonIntrusiveProfile.init()

        mock_dp_init.assert_called_once_with("")
        mock_print_warn.assert_called_once()
        mock_register.assert_called_once_with()

    def test_config_path_trigger_registers_hook(self):
        env = {
            "PROF_CONFIG_PATH": "profiler_config",
            "KINETO_USE_DAEMON": None,
            "MSMONITOR_USE_DAEMON": None,
        }
        with patch.object(none_intrusive_profile.os, "getenv", side_effect=self._getenv(env)), \
             patch.object(none_intrusive_profile.PathManager, "check_input_directory_path") as mock_check_path, \
             patch.object(none_intrusive_profile, "dp_init") as mock_dp_init, \
             patch.object(_NonIntrusiveProfile, "_register_optimizer_step_hook") as mock_register:
            _NonIntrusiveProfile.init()

        mock_check_path.assert_called_once_with("profiler_config")
        mock_dp_init.assert_called_once_with("profiler_config")
        mock_register.assert_called_once_with()


if __name__ == "__main__":
    run_tests()
