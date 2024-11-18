import os
import functools

import torch
import torch_npu

from ..utils.path_manager import PathManager
from .dynamic_profile import _DynamicProfile
from .dynamic_profile import init as dp_init
from .dynamic_profile import step as dp_step
from .analysis.prof_common_func._constant import print_error_msg


__all__ = [

]


if torch.__version__ >= "2.0.0":
    _origin_patch_step_function = torch.optim.Optimizer._patch_step_function
elif torch.__version__ >= "1.8.0":
    _origin_patch_step_function = torch.optim.Optimizer._hook_for_profile


class _NonIntrusiveProfile:
    OPTIMIZER_ID = 0
    PROF_CONFIG_PATH = ""
    STEP_RANGE_ID = None

    @staticmethod
    def step_wrapper(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            _NonIntrusiveProfile.step(*args, **kwargs)
            return out

        return wrapper

    @staticmethod
    def get_prof_config_path():
        _NonIntrusiveProfile.PROF_CONFIG_PATH = os.getenv("PROF_CONFIG_PATH", "")
        if not _NonIntrusiveProfile.PROF_CONFIG_PATH:
            return
        try:
            PathManager.check_input_directory_path(_NonIntrusiveProfile.PROF_CONFIG_PATH)
        except RuntimeError as e:
            print_error_msg(f"The path '{_NonIntrusiveProfile.PROF_CONFIG_PATH}' is invalid, "
                            f"and profiler will not be enabled. Error info is: {str(e)}")
            _NonIntrusiveProfile.PROF_CONFIG_PATH = ""

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
        stream = torch.npu.current_stream()
        if _NonIntrusiveProfile.STEP_RANGE_ID is not None:
            torch.npu.mstx.range_end(_NonIntrusiveProfile.STEP_RANGE_ID)
        _NonIntrusiveProfile.STEP_RANGE_ID = torch.npu.mstx.range_start("step", stream)
        if _NonIntrusiveProfile.PROF_CONFIG_PATH != "":
            dp_step()

    @staticmethod
    def init():
        _NonIntrusiveProfile.get_prof_config_path()
        if _NonIntrusiveProfile.PROF_CONFIG_PATH != "":
            dp_init(_NonIntrusiveProfile.PROF_CONFIG_PATH)
        if torch.__version__ >= "2.0.0":
            torch.optim.Optimizer._patch_step_function = _NonIntrusiveProfile.patch_step_function
        elif torch.__version__ >= "1.8.0":
            torch.optim.Optimizer._hook_for_profile = _NonIntrusiveProfile.patch_step_function
