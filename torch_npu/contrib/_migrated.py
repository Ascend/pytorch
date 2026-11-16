import warnings

# Python-layer contrib modules migrated to taco: name -> impl submodule (relative to contrib.module)
MIGRATED_TO_TACO = {
    "DCNv2": "deform_conv",
    "BiLSTM": "bidirectional_lstm",
    "FusedColorJitter": "fusedcolorjitter",
}

_warned = set()


def warn_migrated(name):
    """Aligns with op-plugin TORCH_WARN_ONCE: each name warns at most once per process."""
    if name in _warned:
        return
    _warned.add(name)
    warnings.warn(
        f"torch_npu.contrib.{name} has been migrated to taco. "
        "Please install torch_npu_custom_ops and import taco after torch_npu "
        "to continue using this API. "
        "See https://gitcode.com/Ascend/taco/blob/master/README.md for details. "
        "This API will be removed from torch_npu in a future version.",
        UserWarning,
        stacklevel=3,
    )
