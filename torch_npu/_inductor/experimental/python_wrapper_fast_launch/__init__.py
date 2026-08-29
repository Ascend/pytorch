from .types import FastLaunchError


def __getattr__(name):
    if name in (
        "BoundFastLaunch",
        "FinalizedFastLaunch",
        "bind_python_wrapper_kernel_fast",
    ):
        from .bind import (
            BoundFastLaunch,
            FinalizedFastLaunch,
            bind_python_wrapper_kernel_fast,
        )

        return {
            "BoundFastLaunch": BoundFastLaunch,
            "FinalizedFastLaunch": FinalizedFastLaunch,
            "bind_python_wrapper_kernel_fast": bind_python_wrapper_kernel_fast,
        }[name]
    raise AttributeError(name)


__all__ = [
    "BoundFastLaunch",
    "FinalizedFastLaunch",
    "FastLaunchError",
    "bind_python_wrapper_kernel_fast",
]
