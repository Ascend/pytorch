from torch_npu.utils import _shape_handling as _impl
from torch_npu.utils._shape_handling import (
    NPUShapeHandling,
    unified_copy,
)

__all__ = ["NPUShapeHandling"]

def patch_dynamo_context():
    if not getattr(_impl._patch_shape_handling, "_is_patched", False):
        _impl._patch_dynamo_context()


def patch_shape_handling():
    if getattr(patch_shape_handling, "_is_patched", False):
        return
    patch_dynamo_context()
    patch_shape_handling._is_patched = True
    _impl._patch_shape_handling._is_patched = True
