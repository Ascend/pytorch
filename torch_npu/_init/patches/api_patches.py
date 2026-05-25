from torch_npu._init.patches.patch_manager import PatchManager


@PatchManager.register_patch("api")
def apply_torch_api_patches():
    from torch_npu.multiprocessing.reductions import _add_reductions_methods
    from torch_npu.utils._module import _apply_module_patch
    from torch_npu.utils._optim import add_optim_method
    from torch_npu.utils.collect_env import _add_collect_env_methods
    from torch_npu.utils.dlpack import _apply_dlpack_patch
    from torch_npu.utils.serialization import _add_serialization_methods
    from torch_npu.utils.storage import _add_storage_methods
    from torch_npu.utils.tensor_methods import _add_tensor_methods

    _add_storage_methods()
    _apply_dlpack_patch()
    _apply_module_patch()
    _add_tensor_methods()
    _add_serialization_methods()
    _add_collect_env_methods()
    add_optim_method()
    _add_reductions_methods()
