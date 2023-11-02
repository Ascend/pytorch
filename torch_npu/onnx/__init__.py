from .wrapper_onnx_ops import add_onnx_ops
from .wrapper_ops_combined import add_ops_combined_for_onnx
from .register_aten_ops_to_onnx import native_layer_norm

__all__ = []


add_onnx_ops()
add_ops_combined_for_onnx()
