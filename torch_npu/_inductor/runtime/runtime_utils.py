from torch._inductor.runtime.runtime_utils import (
    triton_config_to_hashable as _original_triton_config_to_hashable,
)


def triton_config_to_hashable(cfg):
    def _freeze_sequence(value):
        if isinstance(value, (list, tuple)):
            return tuple(_freeze_sequence(item) for item in value)
        return value

    return tuple(
        (key, _freeze_sequence(value))
        for key, value in _original_triton_config_to_hashable(cfg)
    )


def patch_triton_config_to_hashable():
    from torch._inductor.runtime import (
        coordinate_descent_tuner,
        runtime_utils,
        triton_heuristics,
    )

    # These modules import the helper directly, so update their bindings too.
    for module in (runtime_utils, coordinate_descent_tuner, triton_heuristics):
        module.triton_config_to_hashable = triton_config_to_hashable
