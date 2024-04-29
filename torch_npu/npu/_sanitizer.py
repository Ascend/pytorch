import os
import atexit

import torch_npu
import torch_npu.utils._npu_trace as npu_trace
import torch_npu.npu._kernel_check as kernel_check
from torch_npu.utils import print_warn_log


class SanitizerMode:
    KERNEL = 1


class NPUSanitizer:

    def __init__(self):
        self.event_handler = None
        self.kernel_path_manager = None
        self.mode = SanitizerMode.KERNEL
        self.opp_debug_path = os.path.join(os.getcwd(), "opp_debug_path")
        self.opp_debug_kernel_path = os.getenv('ASCEND_OPP_DEBUG_PATH')
        self.enabled = False

    def enable(self):
        success = self.enable_kernel_check()
        if not self.enabled and success:
            torch_npu._C._activate_npu_trace(self.mode)
            self.enabled = True

    def enable_kernel_check(self) -> bool:
        if not self.opp_debug_kernel_path:
            print_warn_log("ASCEND_OPP_DEBUG_PATH is not set! TORCH_NPU_SANITIZER takes no effect!")
            return False
        self.kernel_path_manager = kernel_check.KernelPathManager()
        if not os.path.exists(self.opp_debug_path):
            return False
        self.event_handler = kernel_check.EventHandler()
        npu_trace.register_callback_for_acl_start_execution(
            self.event_handler._handle_acl_start_execution,
            "handle_acl_start_execution"
        )
        npu_trace.register_callback_for_acl_finish_execution(
            self.event_handler._handle_acl_finish_execution,
            "handle_acl_finish_execution"
        )
        return True

    def clear_debug_env(self):
        if self.kernel_path_manager:
            self.kernel_path_manager.clear_debug_env()


def enable_npu_sanitizer():
    npu_sanitizer.enable()


npu_sanitizer = NPUSanitizer()

atexit.register(npu_sanitizer.clear_debug_env)
