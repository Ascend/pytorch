import torch

from torch_npu._init.patches.patch_manager import PatchManager


@PatchManager.register_patch("export")
def apply_export_utils_patch():
    """Skip non-OpNamespace entries exposed through ``torch.ops._dir``."""
    import torch._export.utils as export_utils

    original = export_utils._collect_all_valid_cia_ops_for_namespace
    if getattr(original, "_torch_npu_cia_guard", False):
        return

    def guarded_collect(namespace):
        if isinstance(namespace, str):
            op_namespace = getattr(torch.ops, namespace, None)
        else:
            op_namespace = namespace

        if not isinstance(op_namespace, torch._ops._OpNamespace):
            return set()

        # PyTorch 2.7.1 passes the namespace name to this helper. Newer
        # versions pass the materialized namespace object directly.
        if isinstance(namespace, str):
            return original(namespace)
        return original(op_namespace)

    guarded_collect._torch_npu_cia_guard = True
    export_utils._collect_all_valid_cia_ops_for_namespace = guarded_collect
