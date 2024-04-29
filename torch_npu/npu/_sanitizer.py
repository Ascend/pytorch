import os
import atexit

import torch.cuda._sanitizer as csan
import torch_npu
import torch_npu.utils._npu_trace as npu_trace
import torch_npu.npu._stream_check as stream_check
import torch_npu.npu._kernel_check as kernel_check
from torch_npu.utils import print_warn_log


class SanitizerMode:
    STREAM = 0
    KERNEL = 1


class NPUSanitizer:

    def __init__(self):
        self.event_handler = None
        self.dispatch = None
        self.kernel_path_manager = None
        self.mode = SanitizerMode.STREAM
        self.opp_debug_path = os.path.join(os.getcwd(), "opp_debug_path")
        self.opp_debug_kernel_path = os.getenv('ASCEND_OPP_DEBUG_PATH')
        self.enabled = False

    def enable(self):
        if self.opp_debug_kernel_path:
            success = self.enable_kernel_check()
            self.mode = SanitizerMode.KERNEL
        else:
            success = self.enable_stream_check()
            self.mode = SanitizerMode.STREAM
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

    def enable_stream_check(self) -> bool:
        self.event_handler = csan.EventHandler()
        self.dispatch = stream_check.NPUSanitizerDispatchMode(self.event_handler)
        self.dispatch.__enter__()
        npu_trace.register_callback_for_npu_event_creation(
            self.event_handler._handle_event_creation,
            "handle_event_creation"
        )
        npu_trace.register_callback_for_npu_event_deletion(
            self.event_handler._handle_event_deletion,
            "handle_event_deletion"
        )
        npu_trace.register_callback_for_npu_event_record(
            self.event_handler._handle_event_record,
            "handle_event_record"
        )
        npu_trace.register_callback_for_npu_event_wait(
            self.event_handler._handle_event_wait,
            "handle_event_wait"
        )
        npu_trace.register_callback_for_npu_memory_allocation(
            self.event_handler._handle_memory_allocation,
            "handle_memory_allocation"
        )
        npu_trace.register_callback_for_npu_memory_deallocation(
            self.event_handler._handle_memory_deallocation,
            "handle_memory_deallocation"
        )
        npu_trace.register_callback_for_npu_stream_creation(
            self.event_handler._handle_stream_creation,
            "handle_stream_creation"
        )
        npu_trace.register_callback_for_npu_device_synchronization(
            self.event_handler._handle_device_synchronization,
            "handle_device_synchronization"
        )
        npu_trace.register_callback_for_npu_stream_synchronization(
            self.event_handler._handle_stream_synchronization,
            "handle_stream_synchronization"
        )
        npu_trace.register_callback_for_npu_event_synchronization(
            self.event_handler._handle_event_synchronization,
            "handle_event_synchronization"
        )
        return True

    def __del__(self):
        if self.dispatch:
            self.dispatch.__exit__(None, None, None)

    def clear_debug_env(self):
        if self.kernel_path_manager:
            self.kernel_path_manager.clear_debug_env()


def enable_npu_sanitizer():
    npu_sanitizer.enable()


npu_sanitizer = NPUSanitizer()

atexit.register(npu_sanitizer.clear_debug_env)
