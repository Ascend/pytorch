import logging
from typing import Callable, Generic, List

from typing_extensions import ParamSpec  # Python 3.10+

logger = logging.getLogger(__name__)
P = ParamSpec("P")


class CallbackRegistry(Generic[P]):
    def __init__(self, name: str):
        self.name = name
        self.callback_list = []

    def add_callback(self, cb: Callable[P, None], cb_name: str) -> None:
        self.callback_list.append((cb, cb_name))

    def fire_callbacks(self, *args: P.args, **kwargs: P.kwargs) -> None:
        for cb, cb_name in self.callback_list:
            try:
                cb(*args, **kwargs)
            except Exception as e:
                logger.exception(
                    f"Exception in callback {cb_name} for {self.name} registered with NPU trace"
                )


NPUACLExecuteCallbacks: "CallbackRegistry[str]" = CallbackRegistry(
    "NPU acl execution"
)


def register_callback_for_acl_execution(cb: Callable[[str], None], cb_name) -> None:
    NPUACLExecuteCallbacks.add_callback(cb, cb_name)
