try:
    import torch_mlir
    from torch_mlir import ir
except Exception as error:
    raise ImportError("torch_mlir is not installed, install it first.") from error


def import_npu_inductor_plugin():
    from .ascend_npu_ir.ascend_npu_ir.npu import npu_inductor_plugin


import_npu_inductor_plugin()