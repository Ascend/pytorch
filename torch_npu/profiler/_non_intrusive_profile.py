import os
from weakref import WeakKeyDictionary

from torch.optim.optimizer import Optimizer, register_optimizer_step_post_hook

from ..utils._path_manager import PathManager
from ._dynamic_profiler._dynamic_profiler_utils import DynamicProfilerUtils
from .dynamic_profile import init as dp_init
from .dynamic_profile import step as dp_step
from .analysis.prof_common_func._constant import print_error_msg, print_warn_msg


__all__ = [

]


class _NonIntrusiveProfile:
    _optimizer_step_hook_handle = None
    _optimizer_steps = WeakKeyDictionary()
    _profiler_step = 0

    @classmethod
    def _optimizer_step_post_hook(cls, optimizer: Optimizer, _args: tuple, _kwargs: dict) -> None:
        step = cls._optimizer_steps.get(optimizer, 0) + 1
        cls._optimizer_steps[optimizer] = step
        if step > cls._profiler_step:
            dp_step()
            cls._profiler_step = step

    @classmethod
    def _register_optimizer_step_hook(cls) -> None:
        if cls._optimizer_step_hook_handle is not None:
            return
        cls._optimizer_steps.clear()
        cls._profiler_step = 0
        cls._optimizer_step_hook_handle = register_optimizer_step_post_hook(
            cls._optimizer_step_post_hook
        )

    @staticmethod
    def init():
        prof_config_path = os.getenv("PROF_CONFIG_PATH", "")
        kine_to_value = os.getenv("KINETO_USE_DAEMON")
        msmonitor_value = os.getenv("MSMONITOR_USE_DAEMON")

        if kine_to_value is not None:
            print_warn_msg(
                "Environment variable 'KINETO_USE_DAEMON' will be deprecated. "
                "Please use 'MSMONITOR_USE_DAEMON' instead."
            )
        dyno_enable_flag = msmonitor_value or kine_to_value or 0
        try:
            dyno_enable_flag = int(dyno_enable_flag)
        except ValueError:
            print_error_msg("Environment variable 'MSMONITOR_USE_DAEMON' value not valid, will be set to 0 !")
            dyno_enable_flag = 0
        if not prof_config_path and dyno_enable_flag != 1:
            return
        is_dyno = True
        if prof_config_path:
            try:
                PathManager.check_input_directory_path(prof_config_path)
            except RuntimeError:
                print_error_msg(f"The path '{prof_config_path}' is invalid, and profiler will not be enabled.")
                return
            is_dyno = False
        if is_dyno:
            DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL = DynamicProfilerUtils.DynamicProfilerConfigModel.DYNO_CONFIG
        dp_init(prof_config_path)
        _NonIntrusiveProfile._register_optimizer_step_hook()
