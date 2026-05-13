from torch_npu._init.patches.patch_manager import PatchManager


@PatchManager.register_patch("asd")
def apply_asd_patch():
    from torch_npu.asd.asd import _asd_patch

    _asd_patch()
