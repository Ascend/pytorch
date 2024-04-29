import os
import time
from typing import Callable
from torch_npu.utils import print_error_log


def print_check_msg(msg: str):
    pid = os.getpid()
    print(f"[sanitizer]({pid}) {msg}")


class CallbackRegistry:
    def __init__(self, name: str):
        self.name = name
        self.callback_list = []

    def add_callback(self, cb: Callable, cb_name: str) -> None:
        self.callback_list.append((cb, cb_name))

    def fire_callbacks(self, *args, **kwargs) -> None:
        for cb, cb_name in self.callback_list:
            try:
                cb(*args, **kwargs)
            except Exception as e:
                print_error_log(
                    f"Exception in callback {cb_name} for {self.name} registered with NPU trace"
                )


NPUACLStartExecuteCallbacks: "CallbackRegistry" = CallbackRegistry(
    "[kernel check] NPU acl start execution"
)
NPUACLFinishExecuteCallbacks: "CallbackRegistry" = CallbackRegistry(
    "[kernel check] NPU acl finish execution"
)


def register_callback_for_acl_start_execution(cb: Callable[[str], None], cb_name: str) -> None:
    NPUACLStartExecuteCallbacks.add_callback(cb, cb_name)


def register_callback_for_acl_finish_execution(cb: Callable[[str], None], cb_name: str) -> None:
    NPUACLFinishExecuteCallbacks.add_callback(cb, cb_name)
