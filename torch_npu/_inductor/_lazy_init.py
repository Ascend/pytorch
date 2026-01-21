try:
    import torch_mlir
    from torch_mlir import ir
except Exception as error:
    raise ImportError("torch_mlir is not installed, install it first.") from error
from .ascend_npu_ir.build_ext import build_ascend_npu_ir_ext, set_torch_npu_library_path
_has_inited = False
if not _has_inited:
    _has_inited = True
    build_ascend_npu_ir_ext()
    

def import_npu_inductor_plugin():
    from .ascend_npu_ir.ascend_npu_ir.npu import npu_inductor_plugin


set_torch_npu_library_path()
import_npu_inductor_plugin()