from .types import FastLaunchError


def __getattr__(name):
    if name in ("BoundFastLaunch", "bind_python_wrapper_kernel_fast"):
        from .bind import BoundFastLaunch, bind_python_wrapper_kernel_fast

        return {
            "BoundFastLaunch": BoundFastLaunch,
            "bind_python_wrapper_kernel_fast": bind_python_wrapper_kernel_fast,
        }[name]
    raise AttributeError(name)


__all__ = [
    "BoundFastLaunch",
    "FastLaunchError",
    "bind_python_wrapper_kernel_fast",
]
