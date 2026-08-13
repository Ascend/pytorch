from torch_npu._init.patches.patch_manager import PatchManager


@PatchManager.register_patch("npu")
def apply_npu_format_patch():
    from torch_npu.npu._format import _apply_npu_format_patch

    _apply_npu_format_patch()


@PatchManager.register_patch("npu")
def apply_flex_attention_patch():
    from torch_npu.utils.patch_flexattention import (
        _patch_flex_attention_device,
        _register_npu_flex_attention_autocast,
    )

    _patch_flex_attention_device()
    _register_npu_flex_attention_autocast()
