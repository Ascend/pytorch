import os
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
NPUEventCreationCallbacks: "CallbackRegistry" = CallbackRegistry(
    "[stream check] NPU event creation"
)
NPUEventDeletionCallbacks: "CallbackRegistry" = CallbackRegistry(
    "[stream check] NPU event deletion"
)
NPUEventRecordCallbacks: "CallbackRegistry" = CallbackRegistry(
    "[stream check] NPU event record"
)
NPUEventWaitCallbacks: "CallbackRegistry" = CallbackRegistry(
    "[stream check] NPU event wait"
)
NPUMemoryAllocationCallbacks: "CallbackRegistry" = CallbackRegistry(
    "[stream check] NPU memory allocation"
)
NPUMemoryDeallocationCallbacks: "CallbackRegistry" = CallbackRegistry(
    "[stream check] NPU memory deallocation"
)
NPUStreamCreationCallbacks: "CallbackRegistry" = CallbackRegistry(
    "[stream check] NPU stream creation"
)
NPUDeviceSynchronizationCallbacks: "CallbackRegistry" = CallbackRegistry(
    "[stream check] NPU device synchronization"
)
NPUStreamSynchronizationCallbacks: "CallbackRegistry" = CallbackRegistry(
    "[stream check] NPU stream synchronization"
)
NPUEventSynchronizationCallbacks: "CallbackRegistry" = CallbackRegistry(
    "[stream check] NPU event synchronization"
)


def register_callback_for_acl_start_execution(cb: Callable[[str], None], cb_name: str) -> None:
    NPUACLStartExecuteCallbacks.add_callback(cb, cb_name)


def register_callback_for_acl_finish_execution(cb: Callable[[str], None], cb_name: str) -> None:
    NPUACLFinishExecuteCallbacks.add_callback(cb, cb_name)


def register_callback_for_npu_event_creation(cb: Callable[[int], None], cb_name: str) -> None:
    NPUEventCreationCallbacks.add_callback(cb, cb_name)


def register_callback_for_npu_event_deletion(cb: Callable[[int], None], cb_name: str) -> None:
    NPUEventDeletionCallbacks.add_callback(cb, cb_name)


def register_callback_for_npu_event_record(cb: Callable[[int, int], None], cb_name: str) -> None:
    NPUEventRecordCallbacks.add_callback(cb, cb_name)


def register_callback_for_npu_event_wait(cb: Callable[[int, int], None], cb_name: str) -> None:
    NPUEventWaitCallbacks.add_callback(cb, cb_name)


def register_callback_for_npu_memory_allocation(cb: Callable[[int], None], cb_name: str) -> None:
    NPUMemoryAllocationCallbacks.add_callback(cb, cb_name)


def register_callback_for_npu_memory_deallocation(cb: Callable[[int], None], cb_name: str) -> None:
    NPUMemoryDeallocationCallbacks.add_callback(cb, cb_name)


def register_callback_for_npu_stream_creation(cb: Callable[[int], None], cb_name: str) -> None:
    NPUStreamCreationCallbacks.add_callback(cb, cb_name)


def register_callback_for_npu_device_synchronization(cb: Callable[[], None], cb_name: str) -> None:
    NPUDeviceSynchronizationCallbacks.add_callback(cb, cb_name)


def register_callback_for_npu_stream_synchronization(cb: Callable[[int], None], cb_name: str) -> None:
    NPUStreamSynchronizationCallbacks.add_callback(cb, cb_name)


def register_callback_for_npu_event_synchronization(cb: Callable[[int], None], cb_name: str) -> None:
    NPUEventSynchronizationCallbacks.add_callback(cb, cb_name)
