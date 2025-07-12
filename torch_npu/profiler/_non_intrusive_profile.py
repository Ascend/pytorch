import os
import sys
import functools

import torch

from ..utils._path_manager import PathManager
from ._dynamic_profiler._dynamic_profiler_utils import DynamicProfilerUtils
from .dynamic_profile import init as dp_init
from .dynamic_profile import step as dp_step
from .analysis.prof_common_func._constant import print_error_msg, print_warn_msg


__all__ = [

]


if torch.__version__ >= "2.0.0":
    _origin_patch_step_function = torch.optim.Optimizer._patch_step_function
elif torch.__version__ >= "1.8.0":
    _origin_patch_step_function = torch.optim.Optimizer._hook_for_profile


class _NonIntrusiveProfile:
    OPTIMIZER_ID = 0

    @staticmethod
    def step_wrapper(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            _NonIntrusiveProfile.step(*args, **kwargs)
            return out

        return wrapper

    @staticmethod
    def patch_step_function(optimizer: torch.optim.Optimizer):
        _origin_patch_step_function(optimizer)
        _NonIntrusiveProfile.OPTIMIZER_ID = id(optimizer)  # record the last optimizer
        step_hooked = getattr(optimizer.__class__.step, "step_hooked", None)
        if not step_hooked:
            optimizer.__class__.step = _NonIntrusiveProfile.step_wrapper(optimizer.__class__.step)
            optimizer.__class__.step.step_hooked = True

    @staticmethod
    def check_last_optimizer(optimizer: torch.optim.Optimizer):
        return id(optimizer) == _NonIntrusiveProfile.OPTIMIZER_ID

    @staticmethod
    def step(*args, **kwargs):
        optimizer, *_ = args
        if not _NonIntrusiveProfile.check_last_optimizer(optimizer):
            return
        dp_step()

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
        if is_dyno and sys.version_info < (3, 8):
            print_error_msg(f"Dynolog only supported above Python 3.8 !.")
            return
        elif is_dyno:
            DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL = DynamicProfilerUtils.DynamicProfilerConfigModel.DYNO_CONFIG
        dp_init(prof_config_path)
        if torch.__version__ >= "2.0.0":
            torch.optim.Optimizer._patch_step_function = _NonIntrusiveProfile.patch_step_function
        elif torch.__version__ >= "1.8.0":
            torch.optim.Optimizer._hook_for_profile = _NonIntrusiveProfile.patch_step_function
