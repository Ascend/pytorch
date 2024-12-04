import os
import fcntl
import shutil
import traceback
from typing import Callable
import torch_npu.utils._npu_trace as npu_trace
from torch_npu.utils.utils import _print_info_log, _print_error_log, _print_warn_log


class EventHandler:

    def _handle_acl_start_execution(self, acl_name: str):
        npu_trace.print_check_msg(f"====== Start acl operator {acl_name}")

    def _handle_acl_finish_execution(self, acl_name: str):
        # The stack trace generated in this way is in the inverse order, so it must be
        # reversed.
        stack_trace = traceback.StackSummary.extract(
            traceback.walk_stack(None), lookup_lines=False
        )
        stack_trace.reverse()
        stack_str = "".join(stack_trace.format())
        npu_trace.print_check_msg(f"====== {acl_name} python stack info:\n{stack_str}")


class KernelPathManager:

    def __init__(self):
        self.folder_permission = 0o750
        self.ascend_opp_path = os.getenv('ASCEND_OPP_PATH')
        self.opp_debug_path = os.path.join(os.getcwd(), "opp_debug_path")
        self.opp_debug_kernel_path = os.getenv('ASCEND_OPP_DEBUG_PATH')
        self.func_with_lock(self.make_opp_debug_path)
        os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
        os.environ["ASCEND_OPP_PATH"] = self.opp_debug_path

    def func_with_lock(self, func: Callable):
        lock_file = os.path.join(os.getcwd(), "opp_debug.lock")
        with os.fdopen(os.open(lock_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=0o600), "w") as lf:
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                func()
            except IOError:
                _print_warn_log(f"Permission denied to execute {func}")
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def make_opp_debug_path(self):
        if not os.path.exists(self.ascend_opp_path):
            _print_error_log("ASCEND_OPP_PATH is not exists.")
            return
        if not os.path.exists(self.opp_debug_kernel_path):
            _print_error_log("ASCEND_OPP_DEBUG_PATH is not exists.")
            return

        linked_path = os.path.join(self.opp_debug_kernel_path, "built-in/op_impl/ai_core/tbe/kernel")
        if not os.path.exists(linked_path):
            _print_warn_log("ASCEND_OPP_DEBUG_PATH is not valid kernel path.")
            return
        kernel_path = os.path.join(self.opp_debug_path, "built-in/op_impl/ai_core/tbe/kernel")
        if os.path.exists(kernel_path) and os.path.islink(kernel_path):
            os.unlink(kernel_path)
        if os.path.exists(self.opp_debug_path):
            shutil.rmtree(self.opp_debug_path)
        shutil.copytree(self.ascend_opp_path, self.opp_debug_path)
        os.chmod(self.opp_debug_path, self.folder_permission)
        if os.path.exists(kernel_path):
            if os.path.islink(kernel_path):
                os.unlink(kernel_path)
            else:
                shutil.rmtree(kernel_path)
        os.symlink(linked_path, kernel_path)

    def remove_debug_files(self):
        kernel_path = os.path.join(self.opp_debug_path, "built-in/op_impl/ai_core/tbe/kernel")
        os.unlink(kernel_path)
        shutil.rmtree(self.opp_debug_path)

    def clear_debug_env(self):
        if not os.path.exists(self.opp_debug_path):
            return
        lock_file = os.path.join(os.getcwd(), "opp_debug.lock")
        self.func_with_lock(self.remove_debug_files)
        try:
            os.remove(lock_file)
        except FileNotFoundError as err:
            _print_info_log(err)
