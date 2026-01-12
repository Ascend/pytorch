from torch._inductor import config
from . import ascend_custom_passes


def pre_grad_custom_pass_fuc():
    config.pre_grad_custom_pass = ascend_custom_passes.run_register_pre_custom_passes


def post_grad_custom_pass_fuc():
    config.post_grad_custom_post_pass = ascend_custom_passes.run_register_post_custom_passes