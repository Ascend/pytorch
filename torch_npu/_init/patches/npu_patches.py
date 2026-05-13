from torch_npu._init.patches.patch_manager import PatchManager


@PatchManager.register_patch("npu")
def apply_npu_intercept_patch():
    from torch_npu.utils.npu_intercept import _add_intercept_methods

    _add_intercept_methods()


@PatchManager.register_patch("npu")
def apply_npu_format_patch():
    from torch_npu.npu._format import _apply_npu_format_patch

    _apply_npu_format_patch()


@PatchManager.register_patch("npu")
def apply_npu_meta_patch():
    from torch_npu.utils._npu_meta_registration import npu_patch_meta

    npu_patch_meta()
