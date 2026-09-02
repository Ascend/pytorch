from torch_npu._init.patches.patch_manager import PatchManager


@PatchManager.register_patch("dynamo")
def apply_dynamo_methods_patch():
    from torch_npu.utils._dynamo import add_dynamo_methods

    add_dynamo_methods()
