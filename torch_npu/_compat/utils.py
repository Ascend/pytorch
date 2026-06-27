from torch_npu._compat.version import CURRENT_VERSION


# COMPAT(>= 2.12): _ConfigEntry.__init__ gained a required `name` parameter.
# Always pass `name=` at call sites; this wrapper drops it for older versions.
# CAN REMOVE when MIN_SUPPORTED >= (2, 12): construct _ConfigEntry directly.
def make_config_entry(config, *, name: str):
    from torch.utils._config_module import _ConfigEntry
    if CURRENT_VERSION >= (2, 12):
        return _ConfigEntry(config, name=name)
    return _ConfigEntry(config)  # type: ignore[call-arg] - may missing `name` args for 2.13+ versions
