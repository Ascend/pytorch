import os
import atexit
import shutil
import logging
import traceback
from typing import List, Dict

import torch_npu
import torch_npu.utils._npu_trace as npu_trace


logger = logging.getLogger(__name__)


class EventHandler:

    def _handle_acl_execution(self, acl_name: str):
        stack_trace = traceback.StackSummary.extract(
            traceback.walk_stack(None), lookup_lines=False
        )
        stack_trace.reverse()
        stack_str = "".join(stack_trace.format())
        print(f"====== {acl_name} python stack info:\n{stack_str}")


class NPUSanitizer:

    def __init__(self):
        event_handler = EventHandler()
        npu_trace.register_callback_for_acl_execution(
            event_handler._handle_acl_execution,
            "handle_acl_execution"
        )
        self.folder_permission = 0o750
        self.orig_env_var = {
            "ASCEND_LAUNCH_BLOCKING": os.getenv('ASCEND_LAUNCH_BLOCKING'),
            "ASCEND_OPP_PATH": os.getenv('ASCEND_OPP_PATH')
        }
        self.opp_debug_path = os.path.join(os.getcwd(), "opp_debug_path")
        self.opp_debug_kernel_path = os.getenv('ASCEND_OPP_DEBUG_PATH')
        self.enabled = False

    def enable(self):
        if not self.opp_debug_kernel_path:
            logger.warning("ASCEND_OPP_DEBUG_PATH is not set.")
            return
        linked_path = os.path.join(self.opp_debug_kernel_path, "built-in/op_impl/ai_core/tbe/kernel")
        if not os.path.exists(linked_path):
            logger.warning("ASCEND_OPP_DEBUG_PATH is not valid kernel path.")
            return
        kernel_path = os.path.join(self.opp_debug_path, "built-in/op_impl/ai_core/tbe/kernel")
        if os.path.exists(kernel_path) and os.path.islink(kernel_path):
            os.unlink(kernel_path)
        if os.path.exists(self.opp_debug_path):
            shutil.rmtree(self.opp_debug_path)
        shutil.copytree(self.orig_env_var["ASCEND_OPP_PATH"], self.opp_debug_path)
        os.chmod(self.opp_debug_path, self.folder_permission)
        if os.path.exists(kernel_path):
            if os.path.islink(kernel_path):
                os.unlink(kernel_path)
            else:
                shutil.rmtree(kernel_path)
        os.symlink(linked_path, kernel_path)
        os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
        os.environ["ASCEND_OPP_PATH"] = self.opp_debug_path
        if not self.enabled:
            torch_npu._C._activate_npu_trace()
            self.enabled = True

    def clear_debug_env(self):
        if self.enabled:
            kernel_path = os.path.join(self.opp_debug_path, "built-in/op_impl/ai_core/tbe/kernel")
            os.unlink(kernel_path)
            shutil.rmtree(self.opp_debug_path)


def enable_npu_sanitizer():
    npu_sanitizer.enable()


npu_sanitizer = NPUSanitizer()

atexit.register(npu_sanitizer.clear_debug_env)
