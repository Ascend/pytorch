from torch_npu.utils import _dynamo as _impl
from torch_npu.utils._dynamo import NPUShapeHandling

__all__ = ["NPUShapeHandling"]

unified_copy = _impl.unified_copy


def patch_dynamo_context():
    if not getattr(_impl._patch_shape_handling, "_is_patched", False):
        _impl._patch_dynamo_context()


def patch_shape_handling():
    if getattr(patch_shape_handling, "_is_patched", False):
        return
    patch_dynamo_context()
    patch_shape_handling._is_patched = True
    _impl._patch_shape_handling._is_patched = True
