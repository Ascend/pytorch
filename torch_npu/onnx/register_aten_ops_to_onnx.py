import functools

import torch

from torch.onnx._internal import _beartype, registration
from torch.onnx import symbolic_helper
from torch.onnx.symbolic_opset9 import sub, mul, add, pow, sqrt, reciprocal


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=11)

__all__ = ["native_layer_norm"]


@_onnx_symbolic("aten::native_layer_norm")
@symbolic_helper.quantized_args(True, False, False, False)
@symbolic_helper.parse_args("v", "is", "v", "v", "f")
@_beartype.beartype
def native_layer_norm(g, inputs, normalized_shape, weight, bias, eps):
    axes = [-i for i in range(len(normalized_shape), 0, -1)]

    two_cst = symbolic_helper._generate_wrapped_number(g, 2.0)
    eps_cst = symbolic_helper._generate_wrapped_number(g, eps)

    mean = g.op("ReduceMean", inputs, axes_i=axes)
    numerator = sub(g, inputs, mean)

    # variance = e((x - e(x))^2), and (x - e(x)) is the numerator in the layer_norm formula
    variance = g.op("ReduceMean", pow(g, numerator, two_cst), axes_i=axes)
    denominator = sqrt(g, g.op("Add", variance, eps_cst))
    normalized = g.op("Div", numerator, denominator)

    if not (weight is None or symbolic_helper._is_none(weight)):
        normalized = mul(g, normalized, weight)
    if not (bias is None or symbolic_helper._is_none(bias)):
        normalized = add(g, normalized, bias)

    # rdenominator := 1 / sqrt(variance + eps)
    rdenominator = reciprocal(g, denominator)

    return normalized, mean, rdenominator
