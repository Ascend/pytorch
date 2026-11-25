"""Load and validate the optional GE/OM2 executor when explicitly selected."""

from functools import cache


@cache
def load_backend():
    """Resolve every native symbol before graph compilation; cache only success."""
    from fxrt.gert_backend import backend, check_schema_compatibility

    check_schema_compatibility()
    from fxrt import _fxrt_torch_bridge  # noqa: F401  # validate the ABI=1 bridge too

    return backend
