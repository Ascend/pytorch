from .types import FastLaunchError


def __getattr__(name):
    if name in (
        "BoundFastLaunch",
        "FinalizedFastLaunch",
        "FinalizedStaticLaunch",
        "bind_python_wrapper_kernel_fast",
    ):
        from .bind import (
            BoundFastLaunch,
            FinalizedFastLaunch,
            FinalizedStaticLaunch,
            bind_python_wrapper_kernel_fast,
        )

        return {
            "BoundFastLaunch": BoundFastLaunch,
            "FinalizedFastLaunch": FinalizedFastLaunch,
            "FinalizedStaticLaunch": FinalizedStaticLaunch,
            "bind_python_wrapper_kernel_fast": bind_python_wrapper_kernel_fast,
        }[name]
    raise AttributeError(name)


__all__ = [
    "BoundFastLaunch",
    "FinalizedFastLaunch",
    "FinalizedStaticLaunch",
    "FastLaunchError",
    "bind_python_wrapper_kernel_fast",
]
